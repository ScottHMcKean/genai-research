# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Compare the 3 models
# MAGIC
# MAGIC Aggregates the per-model rows written by `03_scrub_and_redact` (workspace results table) into a
# MAGIC side-by-side view of **quality (F1)** vs **throughput (tokens/s)** vs **latency** — the base Qwen3-4B,
# MAGIC the LoRA-tuned Qwen3-4B, and Gemma-3-27B. Shows whether the small LoRA model closes the quality gap
# MAGIC with the 27B model at a fraction of the serving cost.

# COMMAND ----------

# MAGIC %pip install mlflow==3.12.0 -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
from config import RESULTS_TABLE, REDACTED_TABLE
from pyspark.sql import functions as F, Window

# COMMAND ----------

# Latest run per model_key.
w = Window.partitionBy("model_key").orderBy(F.col("run_ts").desc())
latest = (spark.table(RESULTS_TABLE)
          .withColumn("rn", F.row_number().over(w))
          .filter("rn = 1").drop("rn"))

comparison = latest.select(
    "model_key", "num_docs",
    F.round("micro_f1", 4).alias("micro_f1"),
    F.round("micro_precision", 4).alias("precision"),
    F.round("micro_recall", 4).alias("recall"),
    F.round("macro_f1", 4).alias("macro_f1"),
    F.round("tokens_per_s", 1).alias("tokens_per_s"),
    F.round("p50_ms", 0).alias("p50_ms"),
    F.round("p95_ms", 0).alias("p95_ms"),
    "parse_errors",
).orderBy(F.col("micro_f1").desc())

display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Redacted-output sanity check (shared catalog)
# MAGIC Confirms the shared catalog holds masked text (salt IDs) and no raw values.

# COMMAND ----------

display(
    spark.table(REDACTED_TABLE)
    .select("model_key", "document_type", "num_entities", "masked_text", "entities_meta_json")
    .limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Headline

# COMMAND ----------

import json
rows = {r["model_key"]: r for r in comparison.collect()}
best_f1 = max(rows.values(), key=lambda r: r["micro_f1"])
fastest = max(rows.values(), key=lambda r: r["tokens_per_s"])
print(f"Best F1:   {best_f1['model_key']} @ {best_f1['micro_f1']}")
print(f"Fastest:   {fastest['model_key']} @ {fastest['tokens_per_s']:.0f} tokens/s")

lora, base = rows.get("qwen3-4b-lora"), rows.get("qwen3-4b")
if lora and base:
    print(f"LoRA lift: Qwen3-4B F1 {base['micro_f1']} -> {lora['micro_f1']} "
          f"(+{lora['micro_f1'] - base['micro_f1']:.4f}) after PII fine-tuning")

dbutils.notebook.exit(json.dumps({"best_f1_model": best_f1["model_key"],
                                  "best_f1": float(best_f1["micro_f1"])}))
