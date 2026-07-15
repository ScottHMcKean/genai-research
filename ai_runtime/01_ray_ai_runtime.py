# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · AI Runtime + Ray — distributed claims triage
# MAGIC
# MAGIC Showcases the **Databricks AI Runtime** with **Ray** for distributed AI work. We
# MAGIC fan out an LLM triage call (severity Low/Medium/High) over many claim notes
# MAGIC concurrently against the Foundation Model API — the single-threaded Python driver
# MAGIC is the bottleneck; Ray removes it.
# MAGIC
# MAGIC > **Environment**: Notebook → Base environment → **`5`** (Serverless v5, Ray preinstalled).
# MAGIC >
# MAGIC > **Ray topology.** On **Serverless** we run Ray *locally on the driver* (`ray.init()`)
# MAGIC > — `ray.util.spark.setup_ray_cluster` touches `spark.sparkContext`, which Spark
# MAGIC > Connect blocks. On a **classic multi-node cluster** use `setup_ray_cluster(...)`
# MAGIC > to fan out across workers (commented pattern at the bottom). Same code otherwise.
# MAGIC
# MAGIC This is the reliable AI-Runtime piece — no GPU required.

# COMMAND ----------

# MAGIC %pip install --quiet "ray[default]" openai mlflow
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".")
from config import NOTES_TABLE, RAY_SCORED_TABLE, CHAT_MODEL, LABELS

import time, json
import pandas as pd
from time import perf_counter
import mlflow

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
WORKSPACE_URL = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
BASE_URL = f"{WORKSPACE_URL}/serving-endpoints"
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/ai_runtime_claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Load claim notes to triage

# COMMAND ----------

# Keep the demo snappy: triage a sample. Bump this to show scaling.
N = 300
notes_df = (spark.table(NOTES_TABLE)
            .select("claim_id", "note", "severity")
            .limit(N).toPandas())
print(f"Loaded {len(notes_df)} claim notes to triage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Start Ray on the driver
# MAGIC
# MAGIC `ray.init()` spins up a local scheduler; `num_cpus` is inferred from the driver.
# MAGIC Task-level `num_cpus=0.25` oversubscribes on purpose — FMAPI calls are
# MAGIC network-bound, so 4× more in-flight tasks than cores is free throughput. The real
# MAGIC cap is the endpoint's rate limit.

# COMMAND ----------

import ray
if ray.is_initialized():
    ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)
print(ray.cluster_resources())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Distributed triage task

# COMMAND ----------

TRIAGE_PROMPT = (
    "You are a P&C claims triage assistant. Read the adjuster note and classify the "
    "claim severity as exactly one of: Low, Medium, High.\n"
    "- Low: cosmetic / minor, no injuries, fast close.\n"
    "- Medium: moderate damage or minor injury, needs assessment.\n"
    "- High: total loss, hospitalization, litigation, or uninhabitable dwelling.\n"
    "Respond with ONLY the single word: Low, Medium, or High.\n\nNote: "
)

def _worker_client(token: str, base_url: str):
    """One OpenAI client per Ray worker process, reused across the notes it handles
    (Ray reuses worker processes across tasks) instead of one per note."""
    from openai import OpenAI
    global _WORKER_CLIENT
    try:
        return _WORKER_CLIENT
    except NameError:
        _WORKER_CLIENT = OpenAI(api_key=token, base_url=base_url)
        return _WORKER_CLIENT


@ray.remote(num_cpus=0.25)
def triage(claim_id: str, note: str, token: str, base_url: str, model: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        client = _worker_client(token, base_url)
        resp = client.chat.completions.create(
            model=model, max_tokens=4, temperature=0,
            messages=[{"role": "user", "content": prompt + note}],
        )
        pred = resp.choices[0].message.content.strip().split()[0].capitalize()
        if pred not in ("Low", "Medium", "High"):
            pred = "Medium"
        return {"claim_id": claim_id, "pred_severity": pred, "error": None,
                "latency_s": time.perf_counter() - t0}
    except Exception as e:
        return {"claim_id": claim_id, "pred_severity": None, "error": str(e),
                "latency_s": time.perf_counter() - t0}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Fan out with Ray and measure throughput

# COMMAND ----------

t0 = perf_counter()
futures = [triage.remote(r.claim_id, r.note, TOKEN, BASE_URL, CHAT_MODEL, TRIAGE_PROMPT)
           for r in notes_df.itertuples(index=False)]
results = ray.get(futures)
wall_s = perf_counter() - t0

res_df = pd.DataFrame(results)
n_err = res_df["error"].notna().sum()
print(f"Ray wall: {wall_s:.1f}s | notes/s: {len(res_df)/wall_s:.2f} | errors: {n_err} | "
      f"p50 latency: {res_df['latency_s'].median():.2f}s")

merged = notes_df.merge(res_df, on="claim_id")
# Score accuracy only over notes that actually returned a prediction -- counting
# transport failures (rate-limit/timeout -> pred_severity=None) as wrong would
# understate the model. Report the coverage separately.
scored = merged[merged["pred_severity"].notna()]
acc = (scored["severity"] == scored["pred_severity"]).mean() if len(scored) else float("nan")
print(f"Zero-shot triage accuracy vs labels: {acc:.3f} "
      f"(over {len(scored)}/{len(merged)} notes; {n_err} failed calls excluded)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b · Serial baseline — how much did Ray actually buy us?
# MAGIC
# MAGIC Triage a small slice **serially** (the naive single-threaded driver loop) and
# MAGIC compare per-note time to the Ray fan-out. For network-bound FMAPI calls the speedup
# MAGIC scales with in-flight concurrency — this is the number a customer evaluating Ray
# MAGIC wants to see.

# COMMAND ----------

from openai import OpenAI

BASE_N = min(24, len(notes_df))          # small slice — serial is the slow path
_client = OpenAI(api_key=TOKEN, base_url=BASE_URL)

def _serial_triage(note: str) -> str:
    r = _client.chat.completions.create(
        model=CHAT_MODEL, max_tokens=4, temperature=0,
        messages=[{"role": "user", "content": TRIAGE_PROMPT + note}],
    )
    return r.choices[0].message.content.strip()

t0 = perf_counter()
for note in notes_df["note"].head(BASE_N):
    _serial_triage(note)
serial_s = perf_counter() - t0

serial_per_note = serial_s / BASE_N
ray_per_note = wall_s / len(res_df)
speedup = serial_per_note / ray_per_note if ray_per_note else float("nan")
print(f"Serial: {serial_per_note:.2f}s/note  |  Ray: {ray_per_note:.2f}s/note  "
      f"|  speedup: {speedup:.1f}x on {int(ray.cluster_resources().get('CPU', 1))} CPUs")
print("Same code scales further on a classic multi-node cluster (see appendix).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Ray Data — the same work as a distributed dataset pipeline
# MAGIC
# MAGIC `@ray.remote` (above) is Ray Core — great for ad-hoc fan-out. **Ray Data** is the
# MAGIC higher-level library for batch pipelines: lazy, streamed, and it spills to disk so it
# MAGIC scales past memory. `map_batches` runs the UDF across the cluster with a controlled
# MAGIC `concurrency`. Same triage, expressed as a dataset transform.

# COMMAND ----------

import ray.data

DATA_N = min(60, len(notes_df))          # small slice to demo the API without a second full pass
ds = ray.data.from_pandas(notes_df[["claim_id", "note"]].head(DATA_N))

def score_batch(batch: pd.DataFrame) -> pd.DataFrame:
    from openai import OpenAI
    client = OpenAI(api_key=TOKEN, base_url=BASE_URL)
    preds = []
    for note in batch["note"]:
        try:
            r = client.chat.completions.create(
                model=CHAT_MODEL, max_tokens=4, temperature=0,
                messages=[{"role": "user", "content": TRIAGE_PROMPT + note}])
            p = r.choices[0].message.content.strip().split()[0].capitalize()
            preds.append(p if p in ("Low", "Medium", "High") else "Medium")
        except Exception:
            preds.append("Medium")
    batch = batch.copy()
    batch["pred_severity"] = preds
    return batch

t0 = perf_counter()
ray_data_out = ds.map_batches(score_batch, batch_size=8, concurrency=8, batch_format="pandas").to_pandas()
ray_data_s = perf_counter() - t0
print(f"Ray Data scored {len(ray_data_out)} notes in {ray_data_s:.1f}s via map_batches")
display(ray_data_out.head(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Persist + log the run
# MAGIC
# MAGIC These zero-shot predictions are also the **baseline** the fine-tuned model in
# MAGIC `03_serve_and_eval` is compared against.

# COMMAND ----------

# Persist only rows with a valid prediction (the baseline consumed by notebook 03).
(spark.createDataFrame(scored[["claim_id", "note", "severity", "pred_severity", "latency_s"]])
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(RAY_SCORED_TABLE))

with mlflow.start_run(run_name="ray_zero_shot_triage"):
    mlflow.log_param("model", CHAT_MODEL)
    mlflow.log_param("ray_topology", "driver_local_serverless")
    mlflow.log_param("num_cpus_per_task", 0.25)
    mlflow.log_param("n_notes", len(res_df))
    mlflow.log_metric("wall_s", float(wall_s))
    mlflow.log_metric("notes_per_s", float(len(res_df) / wall_s))
    mlflow.log_metric("p50_latency_s", float(res_df["latency_s"].median()))
    mlflow.log_metric("errors", int(n_err))
    mlflow.log_metric("zero_shot_accuracy", float(acc))
    mlflow.log_metric("serial_s_per_note", float(serial_per_note))
    mlflow.log_metric("ray_s_per_note", float(ray_per_note))
    mlflow.log_metric("ray_speedup_x", float(speedup))

print("Wrote", RAY_SCORED_TABLE)
display(spark.table(RAY_SCORED_TABLE).limit(10))

# COMMAND ----------

ray.shutdown()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix — classic multi-node cluster topology
# MAGIC
# MAGIC On a classic cluster (not serverless), fan out across worker nodes instead of the
# MAGIC driver. Attach this notebook to a cluster and use:
# MAGIC
# MAGIC ```python
# MAGIC from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
# MAGIC setup_ray_cluster(
# MAGIC     min_worker_nodes=0, max_worker_nodes=3,
# MAGIC     num_cpus_worker_node=4, num_gpus_worker_node=0,
# MAGIC )
# MAGIC ray.init(ignore_reinit_error=True)
# MAGIC # ... same @ray.remote tasks ...
# MAGIC shutdown_ray_cluster()
# MAGIC ```
# MAGIC
# MAGIC Everything above the topology line is identical — Ray abstracts driver-local vs
# MAGIC multi-node. See `ray/01_ray_basics_classic_cluster.ipynb` for the cluster pattern.
