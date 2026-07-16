# Databricks notebook source
# MAGIC %md
# MAGIC # Serve a custom HF embedding model on AI Runtime + benchmark tokens/s
# MAGIC
# MAGIC Serves an open-source Hugging Face **embedding** model on Databricks GPU **model serving** and
# MAGIC **benchmarks throughput (tokens/s)**.
# MAGIC
# MAGIC Flow:
# MAGIC 1. Snapshot an open-source embedding model to a UC Volume (no HF pull at serve time).
# MAGIC 2. Wrap it as an MLflow `pyfunc`, log with a signature.
# MAGIC 3. Register to Unity Catalog as an **[express deployment](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments)**
# MAGIC    (`env_pack="databricks_model_serving"`) so serving starts fast and the serve env matches the notebook env.
# MAGIC 4. Deploy to a **GPU model serving** endpoint.
# MAGIC 5. **Benchmark tokens/s** — sweep request batch size × client concurrency; report tokens/s, requests/s,
# MAGIC    rows/s, and latency percentiles.
# MAGIC
# MAGIC > **Run on Serverless GPU compute (A10 recommended).** Express deployments pack the environment +
# MAGIC > artifacts at *registration* time in the serverless notebook. Requires `mlflow>=3.1`, `databricks-sdk>=0.102.0`.
# MAGIC >
# MAGIC > **Why pyfunc and not the Custom LLM serving / vLLM entrypoint path?** Custom LLM Serving only supports
# MAGIC > the `llm/v1/chat` task — it rejects `llm/v1/embeddings` (`InvalidParameterValue ... must be llm/v1/chat`),
# MAGIC > so the vLLM custom-entrypoint route cannot serve an embedding model. Standard GPU model serving with a
# MAGIC > pyfunc is the supported path for embeddings, and express deployment (`env_pack`) still applies.
# MAGIC
# MAGIC Context: medical-case inference — historical ground truth (millions of free-form rows) and small per-case
# MAGIC tables live in Unity Catalog SQL tables. This validates custom-embedding serving throughput before wiring
# MAGIC up retrieval.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up the environment (ensure you are running Serverless GPU with an A10 GPU)

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow>=3.1" sentence-transformers "databricks-sdk>=0.102.0"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from databricks.sdk.service.serving import ServingModelWorkloadType

# Open-source HF embedding model. bge-small is a strong, small general model; swap for a domain /
# clinical embedding model once security-cleared in the system AI catalog.
MODEL_REPO_ID = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384
SNAPSHOT_DIR = "/Volumes/shm_skunkworks_catalog/genai/huggingface/bge_small_en_v1_5"

# Unity Catalog destination for the registered model. Must be writable by you.
UC_MODEL_NAME = "shm_skunkworks_catalog.genai.bge_small_embed"

# Serving endpoint. A small embedding model fits on GPU_SMALL; try GPU_MEDIUM (A10) for headroom.
ENDPOINT_NAME = "shm-bge-small-embed"
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_SMALL
WORKLOAD_SIZE = "Small"
SCALE_TO_ZERO_ENABLED = True

# COMMAND ----------

import os
import time
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download the embedding model to a UC Volume
# MAGIC Snapshot the weights to a Volume so the logged artifact is self-contained — no HF pull at serve time
# MAGIC (also addresses the HF-download security concern).

# COMMAND ----------

from huggingface_hub import snapshot_download

# snapshot_download preserves the canonical HF layout (model.safetensors etc.) so both
# sentence-transformers and transformers can reload it; SentenceTransformer.save() does not.
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
snapshot_download(repo_id=MODEL_REPO_ID, local_dir=SNAPSHOT_DIR)
print(f"Downloaded {MODEL_REPO_ID} -> {SNAPSHOT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Wrap as an MLflow `pyfunc` and log with a signature
# MAGIC Thin `PythonModel` around `sentence-transformers`. Input is a DataFrame with a `text` column; output is
# MAGIC one normalized embedding per row (`normalize_embeddings=True` → cosine similarity is a dot product).

# COMMAND ----------

class EmbeddingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        self.model = SentenceTransformer(context.artifacts["repository"], device=device)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]
        batch_size = (params or {}).get("batch_size", 64)
        embeddings = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(embeddings).tolist()

# COMMAND ----------

input_example = pd.DataFrame({"text": ["chest pain radiating to the left arm, elevated troponin"]})
signature = infer_signature(input_example, [[0.0] * EMBED_DIMS], params={"batch_size": 64})

with mlflow.start_run(run_name="bge_small_embed"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=EmbeddingModel(),
        artifacts={"repository": SNAPSHOT_DIR},
        pip_requirements=["sentence-transformers", "torch", "transformers", "accelerate"],
        input_example=input_example,
        signature=signature,
    )
print("logged:", model_info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Smoke-test the model locally

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(model_info.model_uri)
vecs = loaded.predict(pd.DataFrame({"text": ["shortness of breath", "acute myocardial infarction"]}))
print("dims:", len(vecs[0]), "| cosine(sob, ami):", float(np.dot(vecs[0], vecs[1])))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register to Unity Catalog (express deployment)

# COMMAND ----------

# env_pack packs the env + artifacts at registration time (express deployment) — applies to custom
# pyfunc + GPU endpoints. https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments
model_version = mlflow.register_model(
    model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving"
)
print("registered:", UC_MODEL_NAME, "version", model_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy to a GPU model serving endpoint

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
    ],
)

w = WorkspaceClient()
try:
    w.serving_endpoints.create_and_wait(name=ENDPOINT_NAME, config=config, timeout=timedelta(minutes=40))
except Exception as e:
    print("create failed (may already exist), updating config:", e)
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME, served_entities=config.served_entities, timeout=timedelta(minutes=40)
    )
print("endpoint ready:", ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Query the endpoint

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

def embed_request(texts):
    """One /invocations call; returns (latency_s, list_of_embeddings)."""
    t0 = time.perf_counter()
    r = client.predict(
        endpoint=ENDPOINT_NAME,
        inputs={"dataframe_records": [{"text": t} for t in texts]},
    )
    return time.perf_counter() - t0, r["predictions"]

_, preds = embed_request(["shortness of breath", "acute myocardial infarction"])
print("returned", len(preds), "vectors of", len(preds[0]), "dims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Benchmark tokens/s
# MAGIC
# MAGIC For an embedding endpoint, **tokens/s = input tokens embedded per second** (no output tokens). We count
# MAGIC tokens with the model's own tokenizer, then drive the endpoint at varying **request batch sizes** (rows
# MAGIC per request) and **client concurrency** to find the throughput knee. Metrics per run: tokens/s,
# MAGIC requests/s, rows/s, and latency p50/p95/p99.

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_DIR)

_snippets = [
    "Chest pain radiating to left arm, diaphoresis, elevated troponin.",
    "Productive cough, fever, consolidation on chest x-ray consistent with community-acquired pneumonia.",
    "Polyuria, polydipsia, fasting glucose 14 mmol/L, HbA1c 9.1% — new diagnosis of type 2 diabetes mellitus.",
    "Sudden onset unilateral weakness and slurred speech; CT head negative for hemorrhage; acute ischemic stroke.",
    "Exertional dyspnea, orthopnea, bilateral basal crackles, raised BNP, reduced ejection fraction on echo.",
    "Right lower quadrant pain with rebound tenderness and leukocytosis; ultrasound suggestive of appendicitis.",
    "Palpitations with irregular pulse; ECG confirms atrial fibrillation with rapid ventricular response.",
    "Severe thunderclap headache, photophobia, neck stiffness; awaiting lumbar puncture to rule out meningitis.",
]
CORPUS = _snippets * 512  # ~4k rows
avg_tokens = float(np.mean([len(tokenizer.encode(t, add_special_tokens=True)) for t in _snippets]))
print(f"corpus rows: {len(CORPUS)} | avg tokens/row: {avg_tokens:.1f} | "
      f"total tokens: {int(avg_tokens * len(CORPUS)):,}")

def tokens_in(texts):
    return int(sum(len(tokenizer.encode(t, add_special_tokens=True)) for t in texts))

# COMMAND ----------

# Warm up the endpoint (exclude cold-start from timings).
_ = embed_request(CORPUS[:8])
print("warmup ok")

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor

def run_benchmark(corpus, batch_size, concurrency):
    """Fire the corpus at the endpoint in batch_size-row requests across `concurrency` threads."""
    batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]
    latencies = []

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for lat, _ in ex.map(lambda b: embed_request(b), batches):
            latencies.append(lat)
    wall = time.perf_counter() - wall0

    total_tokens = tokens_in(corpus)
    lat = np.array(latencies)
    return {
        "batch_size": batch_size,
        "concurrency": concurrency,
        "requests": len(batches),
        "rows": len(corpus),
        "wall_s": round(wall, 2),
        "tokens_per_s": round(total_tokens / wall, 1),
        "rows_per_s": round(len(corpus) / wall, 1),
        "req_per_s": round(len(batches) / wall, 2),
        "p50_ms": round(float(np.percentile(lat, 50)) * 1000, 1),
        "p95_ms": round(float(np.percentile(lat, 95)) * 1000, 1),
        "p99_ms": round(float(np.percentile(lat, 99)) * 1000, 1),
    }

results = []
for batch_size in [8, 32, 64]:
    for concurrency in [1, 4, 8]:
        r = run_benchmark(CORPUS, batch_size=batch_size, concurrency=concurrency)
        print(r)
        results.append(r)

bench_df = pd.DataFrame(results).sort_values("tokens_per_s", ascending=False)
display(bench_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Headline result

# COMMAND ----------

import json

best = bench_df.iloc[0]
headline = (
    f"Peak throughput: {best.tokens_per_s:,.0f} tokens/s "
    f"({best.rows_per_s:,.0f} rows/s) at batch_size={int(best.batch_size)}, "
    f"concurrency={int(best.concurrency)} on {WORKLOAD_TYPE}.\n"
    f"Latency at that point: p50 {best.p50_ms} ms, p95 {best.p95_ms} ms, p99 {best.p99_ms} ms."
)
print(headline)

dbutils.notebook.exit(json.dumps({
    "model": MODEL_REPO_ID,
    "peak_tokens_per_s": float(best.tokens_per_s),
    "rows_per_s": float(best.rows_per_s),
    "batch_size": int(best.batch_size),
    "concurrency": int(best.concurrency),
    "p50_ms": float(best.p50_ms), "p95_ms": float(best.p95_ms), "p99_ms": float(best.p99_ms),
    "headline": headline,
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notes
# MAGIC - Embedding throughput is dominated by **request batch size** (GPU utilization) until per-request latency
# MAGIC   starts to hurt — use the sweep to pick the operating point.
# MAGIC - For the **one-time historical corpus** (millions of rows), prefer a Spark `mapInPandas` / pandas-UDF job
# MAGIC   with the model on AIR GPU compute over hammering the endpoint; keep the endpoint for the low-volume,
# MAGIC   live per-case embedding.
# MAGIC - Re-run the sweep on `GPU_MEDIUM` (A10) to quantify the throughput/cost trade-off vs `GPU_SMALL`.
