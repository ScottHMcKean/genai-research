# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy `unsloth/Meta-Llama-3.1-8B-bnb-4bit` on vLLM (A100 Custom LLM Serving)
# MAGIC
# MAGIC Single, self-contained notebook that **logs → deploys → queries → benchmarks** the 4-bit
# MAGIC bitsandbytes Llama-3.1-8B checkpoint on Databricks Model Serving via the **Custom LLM Serving
# MAGIC *entrypoint* route** (`task=llm/v1/chat`) — the supported way to run vLLM on Databricks
# MAGIC (the UI surfaces this as a custom LLM / "express" deployment). Mirrors the pattern in
# MAGIC `pii_serving/02_deploy_endpoints.py`, folded into one notebook and adapted for a bnb-4bit model.
# MAGIC
# MAGIC ### What it does
# MAGIC 1. Downloads the HF weights to local disk (unsloth re-uploads are **ungated** — no HF token needed).
# MAGIC 2. Logs an MLflow `ChatModel` placeholder + the vLLM `entrypoint` command as metadata.
# MAGIC 3. Registers to Unity Catalog with `env_pack="databricks_model_serving"`.
# MAGIC 4. Creates/updates a **GPU_LARGE (A100 80 GB)** serving endpoint.
# MAGIC 5. Queries it (OpenAI-compatible chat) and runs a **light tokens/s benchmark** across concurrency.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - **Custom LLM Serving** enabled: Admin Settings → Previews → *Custom LLM Serving* → On.
# MAGIC - Run on a **serverless GPU** notebook/job (`GPU_1xA10`) — express deployments require the `env_pack`
# MAGIC   to be packed from a **GPU** environment (CUDA torch/vLLM). vLLM + bitsandbytes are `%pip`-installed
# MAGIC   below so `env_pack` captures them; the smoke test is off by default to avoid the AI-Runtime FIPS
# MAGIC   trap when importing vLLM in the notebook (the serving container is a separate, non-FIPS image).
# MAGIC - The **endpoint** provisions A100 (`GPU_LARGE`) Model Serving GPU. (This FEVM's serverless GPU is
# MAGIC   A10/H100 for *notebooks*; the A100 is the Model Serving pool — first deploy is the live A100 check.)
# MAGIC
# MAGIC > ⚠ `env_pack` is packed from this serverless-GPU env, so the packed torch/vLLM are CUDA builds.
# MAGIC > A CPU-packed env_pack crashes the serving container on startup.
# MAGIC >
# MAGIC > ⚠ First A100 endpoint build is slow (often 20–40+ min) and can exceed the SDK `create_and_wait`
# MAGIC > timeout while still `DEPLOYMENT_CREATING` — not an error. We poll the state ourselves.

# COMMAND ----------

# MAGIC %pip install vllm==0.11.2 bitsandbytes==0.48.1 transformers==4.57.6 openai==2.17.0 mlflow==3.12.0 hf_transfer==0.1.9 "huggingface_hub>=0.27" "databricks-sdk>=0.102.0" -q
# MAGIC %restart_python
# MAGIC
# MAGIC # vLLM + bitsandbytes ARE installed here so env_pack captures the CUDA builds (needs serverless GPU).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config (widgets)

# COMMAND ----------

dbutils.widgets.text("catalog", "shm_skunkworks_catalog", "UC catalog")
dbutils.widgets.text("schema", "models", "UC schema")
dbutils.widgets.text("model_name", "llama31_8b_bnb4", "UC registered model name")
dbutils.widgets.text("endpoint", "llama31-8b-bnb4", "Serving endpoint name")
dbutils.widgets.text("hf_repo", "unsloth/Meta-Llama-3.1-8B-bnb-4bit", "HuggingFace repo")
dbutils.widgets.dropdown("workload_type", "GPU_LARGE", ["GPU_LARGE", "GPU_MEDIUM"], "Serving GPU (A100 / A10)")
dbutils.widgets.dropdown("smoke_test", "false", ["true", "false"], "Local vLLM smoke test (needs GPU)")
dbutils.widgets.dropdown("scale_to_zero", "true", ["true", "false"], "Attempt scale-to-zero")
dbutils.widgets.dropdown("deploy", "true", ["true", "false"], "Deploy endpoint (false = validate/smoke only)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint")
HF_REPO = dbutils.widgets.get("hf_repo")
WORKLOAD_TYPE_STR = dbutils.widgets.get("workload_type")
SMOKE_TEST = dbutils.widgets.get("smoke_test") == "true"
SCALE_TO_ZERO = dbutils.widgets.get("scale_to_zero") == "true"
DEPLOY = dbutils.widgets.get("deploy") == "true"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
SERVED_MODEL_NAME = MODEL_NAME            # vLLM --served-model-name
ARTIFACTS_PATH = "model"                  # local dir handed to MLflow artifacts

# vLLM / serving tuning
DTYPE = "bfloat16"                        # A100/A10 are Ampere+ -> bf16 compute for bnb-4bit
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
LOCAL_PORT = 3080                         # serverless GPU notebooks allow ports 3000-3999
SERVING_PORT = 8080                       # Model Serving requires 8080

print(f"uc_model={UC_MODEL_NAME} | endpoint={ENDPOINT_NAME} | workload={WORKLOAD_TYPE_STR}")
print(f"hf_repo={HF_REPO} | smoke_test={SMOKE_TEST} | scale_to_zero={SCALE_TO_ZERO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download weights to local disk
# MAGIC `/Workspace` can't hold large weight files; download to a temp dir on the node.

# COMMAND ----------

import os, tempfile
os.environ["HF_HOME"] = tempfile.mkdtemp()          # serverless default cache (~/.cache) isn't writable
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.chdir(tempfile.mkdtemp())                         # /Workspace can't hold large weights

from huggingface_hub import snapshot_download

# Download straight into the artifacts dir (unsloth bnb-4bit is ~5.6 GB, ungated -> no HF token).
if not os.path.exists(f"{ARTIFACTS_PATH}/config.json"):
    snapshot_download(repo_id=HF_REPO, local_dir=ARTIFACTS_PATH)
print("local weights:", sorted(os.listdir(ARTIFACTS_PATH))[:8], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. vLLM chat entrypoint
# MAGIC bitsandbytes pre-quantized checkpoint → tell vLLM to load/serve it in bnb format.

# COMMAND ----------

def entrypoint(port: int) -> str:
    args = [
        # Unset the FIPS flag so vLLM's model-inspection subprocess doesn't hit
        # "FATAL FIPS SELFTEST FAILURE" (the serving-container equivalent of the
        # os.environ.pop("OPENSSL_FORCE_FIPS_MODE") fix that works here).
        "env", "-u", "OPENSSL_FORCE_FIPS_MODE",
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", ARTIFACTS_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--quantization", "bitsandbytes",      # 4-bit bnb weights
        "--load-format", "bitsandbytes",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--enforce-eager",                     # skip CUDA-graph capture -> faster, more robust cold start
    ]
    return " ".join(args)

print(entrypoint(SERVING_PORT))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Local smoke test (optional — standalone vLLM OpenAI server)
# MAGIC Verifies the entrypoint and bnb weights load before we pay for an endpoint build. Needs a GPU.

# COMMAND ----------

import subprocess, requests, time

ready_local = False
if SMOKE_TEST:
    log = open("process.log", "w")
    subprocess.Popen(["bash", "-lc", entrypoint(LOCAL_PORT)], stdout=log, stderr=subprocess.STDOUT,
                     text=True, start_new_session=True)
    deadline = time.time() + 900          # bnb-4bit load + weight processing can be slow
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).ok:
                ready_local = True; break
        except Exception:
            pass
        time.sleep(5)
    print("vLLM server ready:", ready_local)
    # Always dump the vLLM log tail — this is the only place we see vLLM's own output
    # (express-deployment endpoints don't surface the vLLM subprocess stdout).
    print("".join(open("process.log").readlines()[-80:]))
    # Persist the full vLLM log to a volume so it can be read off-node (notebook stdout
    # isn't returned by jobs get-run-output).
    try:
        import shutil
        _dst = f"/Volumes/{CATALOG}/{SCHEMA}/_diag"
        os.makedirs(_dst, exist_ok=True)
        shutil.copy("process.log", f"{_dst}/vllm_smoke.log")
        print("saved vLLM log ->", f"{_dst}/vllm_smoke.log")
    except Exception as e:
        print("log save failed:", e)
else:
    print("smoke_test=false -> skipping local vLLM load (endpoint still runs on the serving GPU)")

# COMMAND ----------

if ready_local:
    r = requests.post(
        f"http://localhost:{LOCAL_PORT}/v1/chat/completions",
        json={"model": SERVED_MODEL_NAME,
              "messages": [{"role": "user", "content": "In one sentence, what is Databricks?"}],
              "max_tokens": 64, "temperature": 0.2},
        timeout=60)
    print(r.json()["choices"][0]["message"]["content"][:400])

# COMMAND ----------

# MAGIC %sh pkill -f vllm.entrypoints.openai.api_server || true

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log (ChatModel placeholder + entrypoint metadata) → register (env_pack)

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

class LLMModel(ChatModel):
    # Placeholder: serving runs the entrypoint, not predict. ChatModel gives the chat signature UC needs.
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

with mlflow.start_run(run_name=f"deploy_{MODEL_NAME}"):
    model_info = mlflow.pyfunc.log_model(
        name=SERVED_MODEL_NAME,
        python_model=LLMModel(),
        artifacts={"model_dir": ARTIFACTS_PATH},
        metadata={"task": "llm/v1/chat", "entrypoint": entrypoint(SERVING_PORT)},
        extra_pip_requirements=["mlflow==3.12.0"],
    )
print("logged:", model_info.model_uri)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
# Ensure the schema exists via the SDK (serverless GPU AI Runtime has no Spark session).
from databricks.sdk import WorkspaceClient as _WC
try:
    _WC().schemas.create(name=SCHEMA, catalog_name=CATALOG)
    print(f"created schema {CATALOG}.{SCHEMA}")
except Exception as e:
    print(f"schema ensure ({CATALOG}.{SCHEMA}):", str(e)[:120])  # already exists -> fine

# Express deployment: env_pack packs the CUDA vLLM env from this serverless-GPU notebook.
model_version = mlflow.register_model(
    model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving")
print("registered:", UC_MODEL_NAME, "v", model_version.version)

# COMMAND ----------

# Validate-only short-circuit: when deploy=false, stop here (used to smoke-test the vLLM
# command on a cheap A10 before paying for the A100 endpoint build).
if not DEPLOY:
    import json as _json
    dbutils.notebook.exit(_json.dumps({"uc_model": UC_MODEL_NAME, "version": str(model_version.version),
                                       "deployed": False, "smoke_ready": bool(ready_local)}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy the A100 (GPU_LARGE) endpoint
# MAGIC Entrypoint GPU endpoints have historically rejected scale-to-zero; we attempt it and fall back to
# MAGIC fixed concurrency. First build can exceed the SDK wait while still `DEPLOYMENT_CREATING` (not an error).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType)
from datetime import timedelta

WORKLOAD_TYPE = {
    "GPU_LARGE": ServingModelWorkloadType.GPU_LARGE,     # A100 80 GB
    "GPU_MEDIUM": ServingModelWorkloadType.GPU_MEDIUM,   # A10 24 GB
}[WORKLOAD_TYPE_STR]

w = WorkspaceClient()
existing = [e.name for e in w.serving_endpoints.list()]

def deploy(scale_to_zero: bool):
    entities = [ServedEntityInput(
        entity_name=UC_MODEL_NAME, entity_version=str(model_version.version),
        workload_type=WORKLOAD_TYPE, workload_size="Small",
        scale_to_zero_enabled=scale_to_zero,
    )]
    if ENDPOINT_NAME in existing:
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME, served_entities=entities, timeout=timedelta(minutes=50))
    else:
        w.serving_endpoints.create_and_wait(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=entities),
            timeout=timedelta(minutes=50))

scale_to_zero = SCALE_TO_ZERO
try:
    deploy(scale_to_zero)
    print(f"endpoint READY: {ENDPOINT_NAME} (scale_to_zero={scale_to_zero})")
except TimeoutError:
    print(f"still deploying (poll below/UI): {ENDPOINT_NAME} state={w.serving_endpoints.get(ENDPOINT_NAME).state}")
except Exception as e:
    msg = str(e).lower()
    if scale_to_zero and ("autoscal" in msg or "scale_to_zero" in msg or "scale to zero" in msg
                          or "entrypoint" in msg):
        print(f"scale-to-zero rejected -> retrying with fixed concurrency.\n  ({e})")
        scale_to_zero = False
        try:
            deploy(scale_to_zero)
            print(f"endpoint READY: {ENDPOINT_NAME} (scale_to_zero=False, fallback)")
        except TimeoutError:
            print(f"still deploying (poll below/UI): {ENDPOINT_NAME} state={w.serving_endpoints.get(ENDPOINT_NAME).state}")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wait for READY (decoupled from create — polls up to ~45 min)

# COMMAND ----------

deadline = time.time() + 45 * 60
ready = False
while time.time() < deadline:
    s = w.serving_endpoints.get(ENDPOINT_NAME).state
    ready_v = getattr(s.ready, "value", str(s.ready)) if s and s.ready else "?"
    cu_v = getattr(s.config_update, "value", str(s.config_update)) if s and s.config_update else "?"
    print(time.strftime("%T"), "ready=", ready_v, "config_update=", cu_v)
    if ready_v == "READY":
        ready = True; break
    time.sleep(30)
print("endpoint READY:", ready)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Query the endpoint (OpenAI-compatible chat)

# COMMAND ----------

from openai import OpenAI

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
client = OpenAI(api_key=TOKEN, base_url=f"{HOST}/serving-endpoints")

if ready:
    resp = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": "Give me three bullet points on why 4-bit quantization helps LLM serving."}],
        max_tokens=256, temperature=0.3,
    )
    print(resp.choices[0].message.content)
    print("\nusage:", resp.usage)
else:
    print("endpoint not READY in the wait window -- poll the UI, then re-run from here.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Light tokens/s benchmark
# MAGIC Measures **generated (completion) tokens/s** against the live endpoint, sweeping client concurrency.
# MAGIC Each request generates up to `GEN_TOKENS`; throughput = total completion tokens / wall time.

# COMMAND ----------

import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

GEN_TOKENS = 256
PROMPTS = [
    "Explain vector databases to a data engineer in 4 sentences.",
    "Write a short Python function that reverses a linked list.",
    "Summarize the tradeoffs of 4-bit vs 8-bit LLM quantization.",
    "Describe how vLLM paged attention improves throughput.",
    "List five best practices for prompt engineering.",
    "What is Unity Catalog and why does governance matter?",
    "Draft a two-sentence product description for a GPU serving platform.",
    "Explain the difference between latency and throughput for LLMs.",
]

def one_call(prompt):
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=GEN_TOKENS, temperature=0.7,
    )
    dt = time.perf_counter() - t0
    return dt, r.usage.completion_tokens

def run_bench(concurrency, n_requests=16):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        out = list(ex.map(one_call, prompts))
    wall = time.perf_counter() - t0
    lat = np.array([o[0] for o in out])
    gen = int(sum(o[1] for o in out))
    return {"concurrency": concurrency, "requests": n_requests, "gen_tokens": gen,
            "wall_s": round(wall, 2), "gen_tokens_per_s": round(gen / wall, 1),
            "req_per_s": round(n_requests / wall, 2),
            "p50_ms": round(float(np.percentile(lat, 50)) * 1000, 1),
            "p95_ms": round(float(np.percentile(lat, 95)) * 1000, 1)}

results = []
if ready:
    print("warmup:", one_call(PROMPTS[0]))
    for conc in [1, 4, 8]:
        r = run_bench(conc)
        print(r)
        results.append(r)
    bench_df = pd.DataFrame(results).sort_values("gen_tokens_per_s", ascending=False)
    display(bench_df)
else:
    print("skipping benchmark -- endpoint not READY")

# COMMAND ----------

import json
summary = {"uc_model": UC_MODEL_NAME, "version": str(model_version.version),
           "endpoint": ENDPOINT_NAME, "workload": WORKLOAD_TYPE_STR,
           "ready": ready, "scale_to_zero": scale_to_zero}
if results:
    best = max(results, key=lambda r: r["gen_tokens_per_s"])
    summary["peak_gen_tokens_per_s"] = best["gen_tokens_per_s"]
    summary["peak_at_concurrency"] = best["concurrency"]
    print(f"peak {best['gen_tokens_per_s']:,.0f} gen-tokens/s at concurrency={best['concurrency']}; "
          f"p50 {best['p50_ms']}ms p95 {best['p95_ms']}ms")
dbutils.notebook.exit(json.dumps(summary))
