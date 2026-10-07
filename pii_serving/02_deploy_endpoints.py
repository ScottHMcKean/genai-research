# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Deploy a model via the vLLM *entrypoint* route
# MAGIC
# MAGIC Serves one model (selected by the `model_key` widget) on Model Serving using the **Custom LLM Serving
# MAGIC entrypoint** path (`task=llm/v1/chat`) — the supported way to run vLLM on Databricks. Mirrors
# MAGIC `custom_models/hf_chat_serving_air.py`. Run once per model, or via the `pii_serving_job` deploy tasks.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - **Custom LLM Serving** enabled: Admin Settings → Previews → *Custom LLM Serving* → On.
# MAGIC - Weights on the volume: base models from `00_setup_and_data`, the LoRA merge from `01_lora_finetune_qwen`.
# MAGIC - Run on **Serverless GPU**. Entrypoint endpoints use **fixed concurrency** (no scale-to-zero).
# MAGIC
# MAGIC > ⚠ `gemma3-27b` and `qwen3-4b-lora` deploy to **GPU_LARGE (A100 80 GB)** — the first deploy is the live
# MAGIC > check that A100 Model Serving is available on this workspace.

# COMMAND ----------

# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 opencv-python-headless==4.12.* mlflow==3.12.0 hf_transfer==0.1.9 databricks-sdk>=0.102.0 -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
import os, tempfile
os.chdir(tempfile.mkdtemp())   # /Workspace doesn't hold large weight files

from config import (
    MODELS, WORKLOAD_TYPES, VOLUME_PATH, DTYPE, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION,
    LOCAL_PORT, SERVING_PORT, SCALE_TO_ZERO_ENABLED,
)

dbutils.widgets.dropdown("model_key", "qwen3-4b", list(MODELS.keys()), "Model to deploy")
dbutils.widgets.dropdown("smoke_test", "true", ["true", "false"], "Local vLLM smoke test (needs GPU)")
MODEL_KEY = dbutils.widgets.get("model_key")
# Local smoke test loads vLLM in-process -> needs a serverless GPU big enough for the model
# (skip for Gemma-27B unless on H100). Off = 02 runs on plain serverless; the endpoint still runs on GPU.
SMOKE_TEST = dbutils.widgets.get("smoke_test") == "true"
m = MODELS[MODEL_KEY]

UC_MODEL_NAME = m["uc_model"]
ENDPOINT_NAME = m["endpoint"]
SERVED_MODEL_NAME = MODEL_KEY.replace("-", "_")
WORKLOAD_TYPE = WORKLOAD_TYPES[m["workload_type"]]
WEIGHTS_DIR = m.get("weights_dir") or f"{VOLUME_PATH}/{MODEL_KEY}"
ARTIFACTS_PATH = "model"

print(f"model_key={MODEL_KEY} | uc={UC_MODEL_NAME} | endpoint={ENDPOINT_NAME} "
      f"| workload={m['workload_type']} | weights={WEIGHTS_DIR}")

# COMMAND ----------

# Copy weights from the volume snapshot to local disk.
import shutil
assert os.path.exists(f"{WEIGHTS_DIR}/config.json"), (
    f"weights not found at {WEIGHTS_DIR} -- run 00_setup_and_data (base) or 01_lora_finetune_qwen (lora) first")
if not os.path.exists(ARTIFACTS_PATH):
    shutil.copytree(WEIGHTS_DIR, ARTIFACTS_PATH)
print("local weights:", sorted(os.listdir(ARTIFACTS_PATH))[:6], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## vLLM chat entrypoint

# COMMAND ----------

def entrypoint(port: int) -> str:
    args = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", ARTIFACTS_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    ]
    return " ".join(args)

print(entrypoint(SERVING_PORT))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local smoke test (standalone vLLM OpenAI server + guided JSON)

# COMMAND ----------

import subprocess, requests, time

ready = False
if SMOKE_TEST:
    log = open("process.log", "w")
    subprocess.Popen(["bash", "-lc", entrypoint(LOCAL_PORT)], stdout=log, stderr=subprocess.STDOUT,
                     text=True, start_new_session=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            if requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).ok:
                ready = True; break
        except Exception:
            pass
        time.sleep(5)
    print("vLLM server ready:", ready)
    if not ready:
        print("".join(open("process.log").readlines()[-40:]))
else:
    print("smoke_test=false -> skipping local vLLM load (endpoint will run on the serving GPU)")

# COMMAND ----------

from pii_common import build_messages, pii_schema

if ready:
    r = requests.post(
        f"http://localhost:{LOCAL_PORT}/v1/chat/completions",
        json={"model": SERVED_MODEL_NAME,
              "messages": build_messages("Contact John Smith at john@acme.com, SSN 123-45-6789."),
              "max_tokens": 300, "temperature": 0.1,
              "guided_json": pii_schema()},
        timeout=60)
    print(r.json()["choices"][0]["message"]["content"][:400])

# COMMAND ----------

# MAGIC %sh pkill -f vllm.entrypoints.openai.api_server || true

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log (ChatModel placeholder + entrypoint metadata) → register (env_pack) → deploy

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

class LLMModel(ChatModel):
    # Placeholder: serving runs the entrypoint, not predict. ChatModel gives the chat signature UC needs.
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

with mlflow.start_run(run_name=f"deploy_{MODEL_KEY}"):
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
model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME,
                                      env_pack="databricks_model_serving")
print("registered:", UC_MODEL_NAME, "v", model_version.version)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from datetime import timedelta
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

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

# Requested: scale-to-zero. Entrypoint endpoints have historically rejected it -> fall back to fixed.
scale_to_zero = SCALE_TO_ZERO_ENABLED
try:
    deploy(scale_to_zero)
    print(f"endpoint READY: {ENDPOINT_NAME} (scale_to_zero={scale_to_zero})")
except TimeoutError:
    # Entrypoint GPU endpoints can exceed the SDK wait while still DEPLOYMENT_CREATING (not an error).
    print(f"still deploying (poll the UI): {ENDPOINT_NAME} state={w.serving_endpoints.get(ENDPOINT_NAME).state}")
except Exception as e:
    msg = str(e).lower()
    if scale_to_zero and ("autoscal" in msg or "scale_to_zero" in msg or "scale to zero" in msg
                          or "entrypoint" in msg):
        print(f"scale-to-zero rejected for entrypoint endpoint -> retrying with fixed concurrency.\n  ({e})")
        scale_to_zero = False
        try:
            deploy(scale_to_zero)
            print(f"endpoint READY: {ENDPOINT_NAME} (scale_to_zero=False, fallback)")
        except TimeoutError:
            print(f"still deploying (poll the UI): {ENDPOINT_NAME} state={w.serving_endpoints.get(ENDPOINT_NAME).state}")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the endpoint (guided JSON)

# COMMAND ----------

from openai import OpenAI

# Wait for the endpoint to actually serve before smoke-querying (entrypoint GPU endpoints report
# READY after the served entity finishes spinning up; querying earlier returns 404). Don't fail the
# run if it's still coming up -- the endpoint is created and 03_scrub_and_redact will use it.
import time
deadline = time.time() + 30 * 60
ready = False
while time.time() < deadline:
    s = w.serving_endpoints.get(ENDPOINT_NAME).state
    if str(getattr(s, "ready", "")).endswith("READY"):
        ready = True; break
    time.sleep(20)
print(f"endpoint ready-state reached: {ready}")

if ready:
    HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    client = OpenAI(api_key=TOKEN, base_url=f"{HOST}/serving-endpoints")
    try:
        resp = client.chat.completions.create(
            model=ENDPOINT_NAME,
            messages=build_messages("Wire $5k to Maria Garcia, IBAN DE89370400440532013000, DOB 1985-03-03."),
            max_tokens=300, temperature=0.1, extra_body={"guided_json": pii_schema()},
        )
        print(resp.choices[0].message.content)
    except Exception as e:
        print(f"smoke query failed (endpoint may still be warming): {e}")
else:
    print("endpoint not READY within wait window -- poll the UI; 03 can still target it once up.")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({"model_key": MODEL_KEY, "endpoint": ENDPOINT_NAME,
                                  "uc_model": UC_MODEL_NAME, "version": str(model_version.version),
                                  "scale_to_zero": scale_to_zero}))
