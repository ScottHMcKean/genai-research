# Databricks notebook source
# MAGIC %md
# MAGIC # Serve a ~1B chat LLM on AI Runtime (Custom LLM serving) + benchmark tokens/s
# MAGIC
# MAGIC Companion to `hf_embedding_serving_air.py`. Where the embedding example uses a `pyfunc` on standard GPU
# MAGIC serving (embeddings aren't supported by Custom LLM serving), **this one uses the
# MAGIC [Custom LLM serving starter](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)
# MAGIC framework** — vLLM custom entrypoint, `llm/v1/chat` task, **express deployment** (`env_pack`) — which *is*
# MAGIC the supported path for generative chat models.
# MAGIC
# MAGIC We serve an open-source **~1B-class instruct model** (`Qwen/Qwen2.5-1.5B-Instruct`) and **benchmark
# MAGIC generative tokens/s** on a **clinical-note classification** task, so the throughput is comparable in
# MAGIC spirit to the small-embedding-model benchmark.
# MAGIC
# MAGIC > **Run on Serverless GPU compute (A10 recommended).** Requires the Custom LLM serving / express-deployment
# MAGIC > feature enabled on the workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up the environment (ensure you are running Serverless GPU with an A10 GPU)

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# Serving requirements (pinned to the starter notebook's tested versions).
# MAGIC %pip install vllm==0.11.2 transformers==4.57.6 openai==2.17.0 opencv-python-headless==4.12.* mlflow==3.12.0 hf_transfer==0.1.9 databricks-sdk>=0.102.0
# MAGIC %restart_python

# COMMAND ----------

# Set working directory to local disk (/Workspace doesn't support large files).
import os, tempfile
workdir = tempfile.mkdtemp()
os.chdir(workdir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from databricks.sdk.service.serving import ServingModelWorkloadType

# ~1B-class open instruct model (ungated, vLLM-supported).
MODEL_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ARTIFACTS_PATH = "qwen"          # Local directory the weights are downloaded to.
SERVED_MODEL_NAME = "qwen"       # Name vLLM exposes the model under.

# vLLM tuning.
DTYPE = "bfloat16"               # A10 (Ampere) supports bf16 natively.
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85

# Allowlisted ports for Serverless GPU notebooks are 3000-3999. Model Serving requires 8080.
LOCAL_PORT = 3080
SERVING_PORT = 8080

# Unity Catalog destination for the registered model. Must be writable by you.
UC_MODEL_NAME = "shm_skunkworks_catalog.genai.qwen25_1_5b_chat"

# Serving endpoint. GPU_MEDIUM (A10, 24 GB) comfortably fits a 1.5B model.
ENDPOINT_NAME = "shm-qwen25-1-5b-chat"
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_MEDIUM
WORKLOAD_SIZE = "Small"
# Custom LLM serving (entrypoint) endpoints require FIXED concurrency — autoscaling/scale-to-zero
# is not supported ("Served entity with entrypoint does not support autoscaling").
SCALE_TO_ZERO_ENABLED = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download the model

# COMMAND ----------

from huggingface_hub import snapshot_download

snapshot_download(repo_id=MODEL_REPO_ID, local_dir=ARTIFACTS_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the model in the notebook

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

# COMMAND ----------

import subprocess

# Start server in background.
log = open("process.log", "w")
subprocess.Popen(
    ["bash", "-lc", entrypoint(LOCAL_PORT)],
    stdout=log,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
)

# COMMAND ----------

# MAGIC %sh
# MAGIC # Tail logs until vLLM is ready. If this hangs, vLLM startup probably encountered an error.
# MAGIC tail -f process.log | sed -u '/Application startup complete/q'

# COMMAND ----------

import requests

resp = requests.post(
    f"http://localhost:{LOCAL_PORT}/invocations",
    json={"messages": [{"role": "user", "content": "Reply with one word: hello"}], "max_tokens": 8},
)
print(resp.json()["choices"][0]["message"]["content"])

# COMMAND ----------

# MAGIC %sh
# MAGIC # Stop the local server.
# MAGIC pkill -f vllm.entrypoints.openai.api_server

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model with our custom entrypoint

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

# Required placeholder. Serving runs the entrypoint, not python_model.predict.
# ChatModel provides the chat signature UC registration needs.
class LLMModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

model_info = mlflow.pyfunc.log_model(
    name=SERVED_MODEL_NAME,
    python_model=LLMModel(),
    artifacts={
        "model_dir": ARTIFACTS_PATH,
    },
    metadata={
        "task": "llm/v1/chat",
        "entrypoint": entrypoint(SERVING_PORT),
    },
    extra_pip_requirements=[
        "mlflow==3.12.0",
    ],
)
model_info.model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog (express deployment)

# COMMAND ----------

import mlflow

# env_pack is required. Custom LLM Serving depends on Serverless Optimized Deployments.
# https://docs.databricks.com/aws/en/machine-learning/model-serving/serverless-optimized-deployments
model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an endpoint with the model

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from datetime import timedelta
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[
        ServedEntityInput(
            entity_name=UC_MODEL_NAME,
            entity_version=str(model_version.version),
            workload_type=WORKLOAD_TYPE,
            workload_size=WORKLOAD_SIZE,
            scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
        )
    ]
)

w = WorkspaceClient()
w.serving_endpoints.create_and_wait(name=ENDPOINT_NAME, config=config, timeout=timedelta(minutes=40))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the ready endpoint

# COMMAND ----------

from openai import OpenAI

DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=f"{DATABRICKS_HOST}/serving-endpoints")

response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[{"role": "user", "content": "Reply with one word: hello"}],
    max_tokens=8,
)
print(response.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark tokens/s on a clinical-note classification task
# MAGIC
# MAGIC For a **generative** endpoint, tokens/s is the standard serving metric. We report **completion tokens/s**
# MAGIC (generation throughput) and **total tokens/s** (prompt + completion), read from `usage`. The workload is a
# MAGIC constrained classification prompt (short outputs), swept over **client concurrency** (chat requests aren't
# MAGIC batched per call like embeddings).

# COMMAND ----------

import time
import numpy as np
import pandas as pd

LABELS = [
    "Acute Coronary Syndrome", "Pneumonia", "Type 2 Diabetes", "Ischemic Stroke",
    "Heart Failure", "Appendicitis", "Atrial Fibrillation", "Meningitis",
]

_notes = [
    "Chest pain radiating to left arm, diaphoresis, elevated troponin.",
    "Productive cough, fever, consolidation on chest x-ray.",
    "Polyuria, polydipsia, fasting glucose 14 mmol/L, HbA1c 9.1%.",
    "Sudden unilateral weakness and slurred speech; CT head negative for hemorrhage.",
    "Exertional dyspnea, orthopnea, bilateral basal crackles, raised BNP.",
    "Right lower quadrant pain, rebound tenderness, leukocytosis.",
    "Palpitations, irregular pulse; ECG shows irregularly irregular rhythm.",
    "Thunderclap headache, photophobia, neck stiffness.",
]
CORPUS = _notes * 32  # 256 classification requests

def classify_prompt(note: str) -> str:
    return (
        "You are a clinical triage classifier. Classify the note into exactly one of these labels:\n"
        + ", ".join(LABELS)
        + f"\nReply with only the label, nothing else.\n\nNote: {note}"
    )

# COMMAND ----------

# Sanity-check a few classifications.
for note in _notes[:3]:
    r = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": classify_prompt(note)}],
        max_tokens=16, temperature=0,
    )
    print(f"{note[:45]:45s} -> {r.choices[0].message.content.strip()}")

# COMMAND ----------

# Warm up.
_ = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[{"role": "user", "content": classify_prompt(_notes[0])}],
    max_tokens=16, temperature=0,
)
print("warmup ok")

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor

def one_request(note):
    r = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": classify_prompt(note)}],
        max_tokens=16, temperature=0,
    )
    u = r.usage
    return u.prompt_tokens, u.completion_tokens

def run_benchmark(corpus, concurrency):
    prompt_toks, completion_toks = 0, 0
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for p, c in ex.map(one_request, corpus):
            prompt_toks += p
            completion_toks += c
    wall = time.perf_counter() - wall0
    total = prompt_toks + completion_toks
    return {
        "concurrency": concurrency,
        "requests": len(corpus),
        "wall_s": round(wall, 2),
        "completion_tokens_per_s": round(completion_toks / wall, 1),
        "total_tokens_per_s": round(total / wall, 1),
        "req_per_s": round(len(corpus) / wall, 2),
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
    }

results = []
for concurrency in [1, 4, 8, 16]:
    r = run_benchmark(CORPUS, concurrency)
    print(r)
    results.append(r)

bench_df = pd.DataFrame(results).sort_values("total_tokens_per_s", ascending=False)
display(bench_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Headline result

# COMMAND ----------

import json

best = bench_df.iloc[0]
headline = (
    f"Peak throughput: {best.total_tokens_per_s:,.0f} total tokens/s "
    f"({best.completion_tokens_per_s:,.0f} completion tokens/s, {best.req_per_s} req/s) "
    f"at concurrency={int(best.concurrency)} on {WORKLOAD_TYPE} for {MODEL_REPO_ID}."
)
print(headline)

dbutils.notebook.exit(json.dumps({
    "model": MODEL_REPO_ID,
    "completion_tokens_per_s": float(best.completion_tokens_per_s),
    "total_tokens_per_s": float(best.total_tokens_per_s),
    "req_per_s": float(best.req_per_s),
    "concurrency": int(best.concurrency),
    "headline": headline,
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes
# MAGIC - Generative throughput scales with concurrency until the GPU saturates (vLLM continuous batching) — the
# MAGIC   sweep finds that point. Completion tokens/s is the fair cross-model number; total tokens/s also counts
# MAGIC   the (longer) classification prompt.
# MAGIC - Not directly comparable to the embedding model's tokens/s: embeddings are encode-only (no generation) on
# MAGIC   a 33M-param model, while this is autoregressive generation on a 1.5B model. Compare *within* each class.
