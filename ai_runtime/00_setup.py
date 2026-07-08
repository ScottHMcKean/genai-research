# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · AI Runtime demo — setup & train/test split
# MAGIC
# MAGIC Idempotent. Uses the **shared claims spine** (`claims_demo.claim_notes`) — it does
# MAGIC **not** regenerate raw data. Builds a stratified train/test split for the
# MAGIC **claims-triage** task: predict `severity` (Low / Medium / High) from the free-text
# MAGIC adjuster `note`.
# MAGIC
# MAGIC If `claim_notes` is missing, run `fins_data/generate_data.py` first.
# MAGIC
# MAGIC Runs on **serverless** (base environment `4` or `5`).

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, ".")
from config import NOTES_TABLE, TRAIN_TABLE, TEST_TABLE, LABELS, SEED

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Assert the shared spine exists

# COMMAND ----------

if not spark.catalog.tableExists(NOTES_TABLE):
    raise RuntimeError(
        f"{NOTES_TABLE} not found. Run fins_data/generate_data.py first "
        "(it builds the shared claims spine)."
    )
notes = spark.table(NOTES_TABLE).select("claim_id", "note", "severity")
print(f"{NOTES_TABLE}: {notes.count()} rows")
display(notes.groupBy("severity").count().orderBy("severity"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Stratified 80/20 train/test split
# MAGIC
# MAGIC Stratify by `severity` so all three classes appear in both splits despite the
# MAGIC natural class imbalance (High is rare — exactly the triage signal we care about).

# COMMAND ----------

fractions = {lbl: 0.8 for lbl in LABELS}
train = notes.sampleBy("severity", fractions=fractions, seed=SEED)
test = notes.join(train.select("claim_id"), on="claim_id", how="left_anti")

(train.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TRAIN_TABLE))
(test.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TEST_TABLE))

print(f"train -> {TRAIN_TABLE}: {spark.table(TRAIN_TABLE).count()}")
print(f"test  -> {TEST_TABLE}: {spark.table(TEST_TABLE).count()}")
display(spark.table(TRAIN_TABLE).groupBy("severity").count().orderBy("severity"))

# COMMAND ----------

# MAGIC %md
# MAGIC Next: `01_ray_ai_runtime` (distributed triage with Ray) and `02_finetune_lora`
# MAGIC (LoRA fine-tune a small model on this split).
