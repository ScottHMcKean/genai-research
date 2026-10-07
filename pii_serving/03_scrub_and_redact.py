# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Scrub PII & write redacted data to the shared catalog
# MAGIC
# MAGIC Reads raw PII from `shm_skunkworks_catalog.pii.raw_documents`, calls the selected serving endpoint with
# MAGIC **guided-JSON** decoding to extract entities, builds masked text, and writes **only redacted data**
# MAGIC (masked text + entity *types/placeholders*, **no original values**) to `shm_catalog.pii.redacted_documents`.
# MAGIC Quality (P/R/F1) and throughput (tokens/s, latency) are logged to MLflow and the workspace results table.
# MAGIC
# MAGIC > **Governance:** raw PII (`input_text`, `original_value`) never leaves the workspace catalog. The shared
# MAGIC > catalog receives masked text where PII is replaced by salt IDs (e.g. `PERSON_001`). Run once per model.

# COMMAND ----------

# MAGIC %pip install openai==2.17.0 mlflow==3.12.0 databricks-sdk>=0.102.0 -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
import json, time
from datetime import datetime, timezone

from config import MODELS, RAW_TABLE, REDACTED_TABLE, RESULTS_TABLE, EXPERIMENT_NAME
from pii_common import extract_pii, build_masked_text, score_document, SYSTEM_PROMPT, COMPACT_SYSTEM_PROMPT

dbutils.widgets.dropdown("model_key", "qwen3-4b", list(MODELS.keys()), "Model / endpoint")
dbutils.widgets.text("concurrency", "8", "Client concurrency")
MODEL_KEY = dbutils.widgets.get("model_key")
CONCURRENCY = int(dbutils.widgets.get("concurrency"))
ENDPOINT = MODELS[MODEL_KEY]["endpoint"]
# The LoRA model was fine-tuned on the compact prompt -> serve it with the same one; base models
# (Qwen, Gemma) use the full few-shot prompt for best zero-shot quality.
SYS_PROMPT = COMPACT_SYSTEM_PROMPT if "lora" in MODEL_KEY else SYSTEM_PROMPT
print(f"model_key={MODEL_KEY} endpoint={ENDPOINT} concurrency={CONCURRENCY} "
      f"prompt={'compact' if 'lora' in MODEL_KEY else 'full'}")

# COMMAND ----------

import pandas as pd
raw_pdf = spark.table(RAW_TABLE).select(
    "doc_id", "document_type", "input_text", "ground_truth_json").toPandas()
print(f"docs to scrub: {len(raw_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract PII (concurrent, guided JSON) + measure throughput

# COMMAND ----------

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
client = OpenAI(api_key=TOKEN, base_url=f"{HOST}/serving-endpoints")

# warm up
_ = extract_pii(client, ENDPOINT, "warmup: contact a@b.com", max_tokens=64, system_prompt=SYS_PROMPT)

def process(rec):
    t0 = time.perf_counter()
    result, n_tokens = extract_pii(client, ENDPOINT, rec["input_text"], system_prompt=SYS_PROMPT)
    return {"rec": rec, "result": result, "tokens": n_tokens, "latency": time.perf_counter() - t0}

records = raw_pdf.to_dict("records")
t_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    outputs = list(ex.map(process, records))
wall = time.perf_counter() - t_start

total_tokens = sum(o["tokens"] for o in outputs)
tokens_per_s = total_tokens / wall if wall > 0 else 0
lat = sorted(o["latency"] for o in outputs)
p50 = lat[len(lat) // 2]
p95 = lat[int(len(lat) * 0.95)]
print(f"wall={wall:.1f}s tokens/s={tokens_per_s:.0f} p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build redacted rows (no raw PII) + score against ground truth

# COMMAND ----------

redacted_rows, doc_scores, parse_errors = [], [], 0
run_ts = datetime.now(timezone.utc)

for o in outputs:
    rec, result = o["rec"], o["result"]
    gt = json.loads(rec["ground_truth_json"])
    if result is None:
        parse_errors += 1
        predicted, masked, meta = [], rec["input_text"], []  # failed parse -> leave unmasked, flagged
        parse_ok = False
    else:
        predicted = [{"type": e.entity_type.value, "value": e.original_value} for e in result.entities]
        masked = build_masked_text(rec["input_text"], result.entities)
        # SHARED-CATALOG SAFE: keep only type + placeholder, never original_value.
        meta = [{"entity_type": e.entity_type.value, "salt_id": e.salt_id} for e in result.entities]
        parse_ok = True

    doc_scores.append(score_document(predicted, gt))
    redacted_rows.append({
        "doc_id": int(rec["doc_id"]),
        "model_key": MODEL_KEY,
        "run_ts": run_ts,
        "document_type": rec["document_type"],
        "masked_text": masked,                    # PII replaced by salt IDs
        "entities_meta_json": json.dumps(meta),   # types + placeholders only
        "num_entities": len(meta),
        "parse_ok": parse_ok,
    })

scores = pd.DataFrame(doc_scores)
tp, fp, fn = int(scores.tp.sum()), int(scores.fp.sum()), int(scores.fn.sum())
micro_p = tp / (tp + fp) if (tp + fp) else 0.0
micro_r = tp / (tp + fn) if (tp + fn) else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
print(f"parse_ok={len(outputs)-parse_errors}/{len(outputs)} "
      f"micro P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write redacted data → `shm_catalog.pii.redacted_documents` (shared metastore)

# COMMAND ----------

red_sdf = spark.createDataFrame(pd.DataFrame(redacted_rows))
(red_sdf.write.mode("overwrite")
        .option("replaceWhere", f"model_key = '{MODEL_KEY}'")
        .option("mergeSchema", "true")
        .saveAsTable(REDACTED_TABLE))
print(f"wrote {red_sdf.count()} redacted rows -> {REDACTED_TABLE} (model_key={MODEL_KEY})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log metrics → MLflow + workspace results table

# COMMAND ----------

import mlflow
_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Users/{_user}/{EXPERIMENT_NAME}")

with mlflow.start_run(run_name=f"scrub_{MODEL_KEY}"):
    mlflow.log_params({"model_key": MODEL_KEY, "endpoint": ENDPOINT, "concurrency": CONCURRENCY,
                       "num_docs": len(outputs)})
    mlflow.log_metrics({
        "tokens_per_s": round(tokens_per_s, 1), "wall_s": round(wall, 2),
        "p50_ms": round(p50 * 1000, 1), "p95_ms": round(p95 * 1000, 1),
        "parse_errors": parse_errors, "tp": tp, "fp": fp, "fn": fn,
        "micro_precision": round(micro_p, 4), "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(float(scores.f1.mean()), 4),
    })

results_row = pd.DataFrame([{
    "run_ts": run_ts, "model_key": MODEL_KEY, "endpoint": ENDPOINT, "num_docs": len(outputs),
    "concurrency": CONCURRENCY, "tokens_per_s": float(tokens_per_s), "wall_s": float(wall),
    "p50_ms": float(p50 * 1000), "p95_ms": float(p95 * 1000), "parse_errors": parse_errors,
    "micro_precision": float(micro_p), "micro_recall": float(micro_r), "micro_f1": float(micro_f1),
    "macro_f1": float(scores.f1.mean()),
}])
spark.createDataFrame(results_row).write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE)
print("logged to MLflow +", RESULTS_TABLE)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"model_key": MODEL_KEY, "micro_f1": round(micro_f1, 4),
                                  "tokens_per_s": round(tokens_per_s, 1),
                                  "redacted_table": REDACTED_TABLE}))
