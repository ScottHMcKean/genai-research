# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# ///

# MAGIC %md
# MAGIC # 01 · Baseline orchestration rollouts on AI Runtime + Ray
# MAGIC
# MAGIC Runs the **policy model** over the full banking-orchestration task set and scores
# MAGIC every rollout with the compliance reward from
# MAGIC `resources_servers/banking_orchestration/app.py` — the baseline GRPO improves on
# MAGIC in notebook 02. Runs on **AI Runtime serverless GPU**; submit with a GPU
# MAGIC accelerator (see repo README — `compute.hardware_accelerator`).
# MAGIC
# MAGIC ## Two upstream blockers on this FEVM (both verified, 2026-08-05)
# MAGIC
# MAGIC **1. NeMo Gym cannot be installed.** Upstream now requires Python `>=3.13.14`;
# MAGIC AI Runtime v5 is Python 3.12.3, so `pip install nemo-gym` fails at resolve time
# MAGIC (`requires a different Python`). No flag works around it. The verifier logic below
# MAGIC is therefore inlined — reward math identical to `app.py`, only NeMo Gym's
# MAGIC serving/CLI plumbing is bypassed.
# MAGIC
# MAGIC **2. vLLM does not run on this image.** Three paths all fail on 1xA10:
# MAGIC - HTTP server subprocess → `SIGABRT`, `crypto/fips/fips.c:154: OpenSSL internal
# MAGIC   error: FATAL FIPS SELFTEST FAILURE` (FIPS-enabled image aborts the child process;
# MAGIC   setting `OPENSSL_CONF=/dev/null` does not help).
# MAGIC - In-process `vllm.LLM` engine → kills the Python kernel (A10G host RAM is ~16GB).
# MAGIC
# MAGIC So this notebook uses **`transformers`** for generation, the same approach that is
# MAGIC verified green in notebook 03. Revisit vLLM on an 8xH100 node, where the in-process
# MAGIC engine has the host RAM to load (the FIPS abort still rules out the HTTP server).

# COMMAND ----------

# MAGIC %pip install --quiet torch transformers accelerate ray mlflow
# MAGIC %restart_python

# COMMAND ----------

import torch, ray
assert torch.cuda.is_available(), "no GPU — submit with compute.hardware_accelerator=GPU_1xA10"
GPU = torch.cuda.get_device_name(0)
print("GPU:", GPU)

# Ray is the rollout fan-out for the scaled (8xH100) configuration, where each worker
# holds its own policy replica. On a single-GPU node there is nothing to fan out to, so
# the rollouts below are batched through one local model instead — same rollouts, same
# rewards, one device. `cluster_resources()` confirms the runtime is up for nb 02.
ray.init(ignore_reinit_error=True, include_dashboard=False)
print("Ray resources:", ray.cluster_resources())

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B" if "A10" in GPU else "Qwen/Qwen3-4B"
print("policy model:", MODEL)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"                     # left-pad so generated tokens align at the tail
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("loaded on", next(model.parameters()).device)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compliance verifier (same reward as `app.py`)
# MAGIC Rule-based, deterministic, auditable:
# MAGIC `0.5·resolved + 0.4·policy_compliant + 0.1·efficiency − 0.5·violations`, clamped `[0,1]`.

# COMMAND ----------

R_RESOLVED, R_POLICY, R_EFFICIENCY, PENALTY = 0.5, 0.4, 0.1, 0.5
ACCOUNT_MODIFYING = {"open_dispute", "block_card", "transfer_funds"}

def score(orchestration, names):
    """Reward for one rollout; `names` is the ordered list of agents the model called."""
    required = set(orchestration["required_agents"])
    intent = orchestration["intent"]
    v = 0
    if any(n in ACCOUNT_MODIFYING for n in names):
        first_mod = min(names.index(n) for n in names if n in ACCOUNT_MODIFYING)
        if "verify_identity" not in names or names.index("verify_identity") > first_mod:
            v += 1
    if "open_dispute" in names:
        if "check_fraud_flags" not in names or names.index("check_fraud_flags") > names.index("open_dispute"):
            v += 1
    if intent == "lost_card" and "block_card" not in names:
        v += 1
    resolved = required.issubset(set(names))
    extras = [n for n in names if n not in required and n != "verify_identity"]
    eff = max(0.0, 1.0 - 0.25 * len(extras))
    r = R_RESOLVED * resolved + R_POLICY * (v == 0) + R_EFFICIENCY * eff - PENALTY * v
    return {"reward": max(0.0, min(1.0, r)), "resolved": int(resolved),
            "violations": v, "efficiency": round(eff, 2), "calls": names}

# COMMAND ----------

import os, re, json

# Task records ship next to this notebook in the repo checkout.
CANDIDATES = [
    os.path.abspath(os.path.join(os.getcwd(), "..",
        "resources_servers/banking_orchestration/data/example.jsonl")),
    ("/Workspace/Users/scott.mckean@databricks.com/genai-research/ray/"
     "03_rl_slm_orchestration/resources_servers/banking_orchestration/data/example.jsonl"),
]
DATA = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert DATA, f"task data not found; looked in {CANDIDATES}"
records = [json.loads(l) for l in open(DATA) if l.strip()]
print(f"{len(records)} task records from {DATA}")

def to_chat_tools(rcp_tools):
    """Responses-API tool defs -> Chat-Completions defs the Qwen chat template expects."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t["parameters"]}}
            for t in rcp_tools]

def tool_names(text):
    """Ordered tool calls Qwen emits as <tool_call>{...}</tool_call> blocks."""
    names = []
    for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try:
            names.append(json.loads(m)["name"])
        except Exception:
            pass
    return names

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the rollouts
# MAGIC Greedy decoding (`do_sample=False`) so the baseline is reproducible; GRPO in nb 02
# MAGIC samples with `temperature=1.0` for exploration.

# COMMAND ----------

BATCH = 8 if "A10" in GPU else 16
prompts, metas = [], []
for rec in records:
    rcp = rec["responses_create_params"]
    msgs = [{"role": "system" if m["role"] == "developer" else m["role"],
             "content": m["content"]} for m in rcp["input"]]
    prompts.append(tok.apply_chat_template(
        msgs, tools=to_chat_tools(rcp["tools"]),
        add_generation_prompt=True, enable_thinking=False, tokenize=False))
    metas.append(rec["extra_info"]["orchestration"])

results = []
for i in range(0, len(prompts), BATCH):
    chunk, chunk_meta = prompts[i:i + BATCH], metas[i:i + BATCH]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]          # left-padded, so slice off the prompt
    for meta, row in zip(chunk_meta, gen):
        text = tok.decode(row, skip_special_tokens=True)
        s = score(meta, tool_names(text))
        s["intent"] = meta["intent"]
        results.append(s)
    print(f"  rollouts {i + len(chunk)}/{len(prompts)}")

# COMMAND ----------

import statistics
from collections import defaultdict

mean_r = statistics.mean(r["reward"] for r in results)
mean_v = statistics.mean(r["violations"] for r in results)
resolved_rate = statistics.mean(r["resolved"] for r in results)

by_intent = defaultdict(list)
for r in results:
    by_intent[r["intent"]].append(r)
for intent, rs in sorted(by_intent.items()):
    print(f"{intent:20s} n={len(rs):3d} reward={statistics.mean(x['reward'] for x in rs):.3f} "
          f"viol={statistics.mean(x['violations'] for x in rs):.2f} "
          f"resolved={statistics.mean(x['resolved'] for x in rs):.2f}")

print(f"\nBASELINE over {len(results)} rollouts ({MODEL}, untrained):")
print(f"  mean_reward    = {mean_r:.3f}")
print(f"  mean_violations= {mean_v:.2f}")
print(f"  resolved_rate  = {resolved_rate:.2f}")
print("GRPO (notebook 02) optimizes exactly this reward.")

# COMMAND ----------

import mlflow
mlflow.set_experiment("/Users/scott.mckean@databricks.com/rl_slm_orchestration")
with mlflow.start_run(run_name=f"baseline_{MODEL.split('/')[-1]}_full"):
    mlflow.log_params({"model": MODEL, "gpu": GPU, "generation": "transformers_greedy",
                       "n_rollouts": len(results), "phase": "baseline_untrained",
                       "data": os.path.basename(DATA)})
    mlflow.log_metric("mean_reward", mean_r)
    mlflow.log_metric("mean_violations", mean_v)
    mlflow.log_metric("resolved_rate", resolved_rate)
    for intent, rs in by_intent.items():
        mlflow.log_metric(f"reward__{intent}", statistics.mean(x["reward"] for x in rs))
        mlflow.log_metric(f"violations__{intent}", statistics.mean(x["violations"] for x in rs))
    mlflow.log_dict({"per_rollout": results}, "baseline_rollouts.json")
print("logged baseline to MLflow")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "gpu": GPU, "model": MODEL, "n_rollouts": len(results),
    "mean_reward": round(mean_r, 3), "mean_violations": round(mean_v, 2),
    "resolved_rate": round(resolved_rate, 2),
    "per_intent": {k: round(statistics.mean(x["reward"] for x in v), 3)
                   for k, v in by_intent.items()},
}))
