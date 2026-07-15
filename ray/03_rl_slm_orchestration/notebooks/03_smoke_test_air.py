# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# ///

# MAGIC %md
# MAGIC # 03 · Smoke test on AI Runtime (A10)
# MAGIC Loads Qwen3 with **transformers** on the serverless GPU (lighter than vLLM for
# MAGIC the A10G's small host RAM), runs our banking tasks through it (tool-calling), and
# MAGIC scores the tool calls with the compliance verifier — proving AI Runtime GPU +
# MAGIC policy model + tool-calling + reward end-to-end.

# COMMAND ----------

# MAGIC %pip install --quiet torch transformers accelerate
# MAGIC %restart_python

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
assert torch.cuda.is_available(), "no GPU — submit with compute.hardware_accelerator=GPU_1xA10"
print("GPU:", torch.cuda.get_device_name(0))

MODEL = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded on", next(model.parameters()).device)

# COMMAND ----------

import re, json, statistics
R_RESOLVED, R_POLICY, R_EFFICIENCY, PENALTY = 0.5, 0.4, 0.1, 0.5
ACCOUNT_MODIFYING = {"open_dispute", "block_card", "transfer_funds"}

def score(meta, names):
    required = set(meta["required_agents"]); intent = meta["intent"]; v = 0
    if any(n in ACCOUNT_MODIFYING for n in names):
        fm = min(names.index(n) for n in names if n in ACCOUNT_MODIFYING)
        if "verify_identity" not in names or names.index("verify_identity") > fm: v += 1
    if "open_dispute" in names:
        if "check_fraud_flags" not in names or names.index("check_fraud_flags") > names.index("open_dispute"): v += 1
    if intent == "lost_card" and "block_card" not in names: v += 1
    resolved = required.issubset(set(names))
    extras = [n for n in names if n not in required and n != "verify_identity"]
    eff = max(0.0, 1.0 - 0.25*len(extras))
    r = R_RESOLVED*resolved + R_POLICY*(v==0) + R_EFFICIENCY*eff - PENALTY*v
    return {"reward": max(0.0, min(1.0, r)), "resolved": int(resolved), "violations": v, "eff": round(eff,2)}

def fn(name, props, desc):
    return {"type":"function","function":{"name":name,"description":desc,
            "parameters":{"type":"object","properties":{k:{"type":t} for k,t in props.items()},
            "required":list(props),"additionalProperties":False}}}
TOOLS = [
    fn("verify_identity",{"customer_id":"string"},"KYC/identity agent"),
    fn("check_fraud_flags",{"account_id":"string"},"Fraud agent"),
    fn("get_balance",{"account_id":"string"},"Accounts agent"),
    fn("open_dispute",{"txn_id":"string","reason":"string"},"Disputes agent"),
    fn("block_card",{"card_id":"string"},"Card-services agent"),
    fn("transfer_funds",{"from_account":"string","to_account":"string","amount":"number"},"Payments agent"),
    fn("escalate_to_human",{"reason":"string"},"Escalation agent"),
]
SYS = ("You are a retail-bank servicing ORCHESTRATOR. Resolve the request by calling "
       "specialist-agent tools. Policy: (1) verify_identity before any account-modifying "
       "action; (2) check_fraud_flags before open_dispute; (3) lost/stolen card must block_card. "
       "Call tools; use the fewest correct steps.")
TASKS = [
    ("I lost my debit card.", {"intent":"lost_card","required_agents":["block_card"]}),
    ("I want to dispute a $250 charge from QuickMart.", {"intent":"dispute_transaction","required_agents":["check_fraud_flags","open_dispute"]}),
    ("What's my checking balance?", {"intent":"balance_inquiry","required_agents":["get_balance"]}),
    ("Move $40 from checking to savings.", {"intent":"transfer_funds","required_agents":["transfer_funds"]}),
]

def tool_names(text):
    names = []
    for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        try: names.append(json.loads(m)["name"])
        except Exception: pass
    return names

# COMMAND ----------

results = []
for utter, meta in TASKS:
    msgs = [{"role":"system","content":SYS},{"role":"user","content":utter}]
    enc = tok.apply_chat_template(
        msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    names = tool_names(text)
    s = score(meta, names)
    results.append(s)
    print(f"{meta['intent']:20s} calls={names} -> {s}")

mean_r = statistics.mean(r["reward"] for r in results)
print(f"\nBASELINE mean reward = {mean_r:.3f}  (untrained {MODEL}; GRPO in nb 02 optimizes this)")
print("SMOKE TEST OK — AI Runtime GPU + Qwen3 tool-calling + compliance reward all ran.")

import mlflow
mlflow.set_experiment("/Users/scott.mckean@databricks.com/rl_slm_orchestration")
with mlflow.start_run(run_name="baseline_qwen3_1.7b"):
    mlflow.log_params({"model": MODEL, "gpu": torch.cuda.get_device_name(0),
                       "n_tasks": len(TASKS), "phase": "baseline_untrained"})
    mlflow.log_metric("mean_reward", mean_r)
    mlflow.log_metric("mean_violations", statistics.mean(r["violations"] for r in results))
    mlflow.log_metric("resolved_rate", statistics.mean(r["resolved"] for r in results))
    for (utter, meta), r in zip(TASKS, results):
        mlflow.log_metric(f"reward__{meta['intent']}", r["reward"])
        mlflow.log_metric(f"violations__{meta['intent']}", r["violations"])
    print("logged baseline to MLflow experiment /Users/scott.mckean@databricks.com/rl_slm_orchestration")

import json as _json
summary = {"gpu": torch.cuda.get_device_name(0), "model": MODEL,
           "mean_reward": round(mean_r, 3),
           "per_task": [{"intent": m[1]["intent"], **r} for m, r in zip(TASKS, results)]}
dbutils.notebook.exit(_json.dumps(summary))
