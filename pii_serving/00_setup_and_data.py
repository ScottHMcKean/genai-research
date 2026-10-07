# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup & data landing
# MAGIC
# MAGIC Prepares the PII-scrubbing demo on **shm-skunkworks**:
# MAGIC 1. Creates the `pii` schema in **both** catalogs + the HuggingFace weight-cache volume.
# MAGIC 2. Verifies the HF token secret (Gemma-3-27B is license-gated).
# MAGIC 3. Streams the gretel financial-PII dataset (English subset) and lands **raw PII** into
# MAGIC    `shm_skunkworks_catalog.pii.raw_documents` (workspace-local, restricted).
# MAGIC 4. Snapshots the base model weights to the UC Volume (download once, reuse).
# MAGIC
# MAGIC > **Governance:** raw PII stays in the workspace catalog. Only redacted data (written by
# MAGIC > `03_scrub_and_redact`) is allowed into the shared metastore catalog `shm_catalog`.
# MAGIC >
# MAGIC > Run on **serverless** (CPU is fine for landing data; weight download is I/O-bound).

# COMMAND ----------

# MAGIC %pip install datasets==3.2.0 huggingface_hub==0.27.0 hf_transfer==0.1.9 -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
import os, tempfile

# Serverless: default HF cache (~/.cache) isn't writable; point it at a temp dir before datasets import.
os.environ["HF_HOME"] = tempfile.mkdtemp()
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from config import (
    CATALOG_RAW, CATALOG_REDACTED, SCHEMA, RAW_TABLE, VOLUME, VOLUME_PATH,
    HF_SECRET_SCOPE, HF_SECRET_KEY, GRETEL_DATASET, NUM_ROWS, MAX_DOC_CHARS, MODELS,
)
from pii_common import parse_ground_truth

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Schemas + volume

# COMMAND ----------

# pii schema in the workspace catalog (raw) + the HF weight-cache volume.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_RAW}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG_RAW}.{SCHEMA}.{VOLUME}")

# pii schema in the shared metastore catalog (redacted output lands here in notebook 03).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_REDACTED}.{SCHEMA}")

print("schemas ready:", f"{CATALOG_RAW}.{SCHEMA}", "|", f"{CATALOG_REDACTED}.{SCHEMA}")
print("weight cache volume:", VOLUME_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. HF token secret (required for the gated Gemma model)

# COMMAND ----------

try:
    hf_token = dbutils.secrets.get(HF_SECRET_SCOPE, HF_SECRET_KEY)
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print(f"HF token loaded from secret {HF_SECRET_SCOPE}/{HF_SECRET_KEY}")
except Exception as e:
    print(f"WARNING: no HF token secret ({HF_SECRET_SCOPE}/{HF_SECRET_KEY}). "
          f"Ungated models (Qwen) will download, but google/gemma-3-27b-it will 401.\n"
          f"  Set it with: databricks secrets put-secret {HF_SECRET_SCOPE} {HF_SECRET_KEY}\n"
          f"  (accept the license at https://huggingface.co/google/gemma-3-27b-it first)\n{e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Land raw PII → `shm_skunkworks_catalog.pii.raw_documents`

# COMMAND ----------

from datasets import load_dataset
import pandas as pd
import json

ds = load_dataset(GRETEL_DATASET, split="train", streaming=True)

rows = []
for row in ds:
    if row["language"] == "English" and len(row["generated_text"]) < MAX_DOC_CHARS:
        text = row["generated_text"]
        gt = parse_ground_truth(row["pii_spans"], text)
        rows.append({
            "doc_id": len(rows),
            "document_type": row["document_type"],
            "input_text": text,
            "pii_spans": str(row["pii_spans"]),
            "ground_truth_json": json.dumps(gt),
            "gt_count": len(gt),
        })
    if len(rows) >= NUM_ROWS:
        break

pii_pdf = pd.DataFrame(rows)
print(f"docs={len(pii_pdf)} types={pii_pdf['document_type'].nunique()} "
      f"gt_entities={pii_pdf['gt_count'].sum()} avg_per_doc={pii_pdf['gt_count'].mean():.1f}")

# COMMAND ----------

(spark.createDataFrame(pii_pdf)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(RAW_TABLE))
print(f"landed {spark.table(RAW_TABLE).count()} rows -> {RAW_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Snapshot base model weights to the volume
# MAGIC
# MAGIC Downloads to local disk first, then copies to the volume (FUSE-mounted volumes don't like
# MAGIC HuggingFace's atomic renames). The LoRA model's weights are produced by `01_lora_finetune_qwen`.

# COMMAND ----------

import shutil
from huggingface_hub import snapshot_download

# Which base models to cache now. Default "all"; pass a comma-separated subset (e.g. "qwen3-4b")
# to cache just one -- useful to validate the cheap path before pulling Gemma-27B (~54 GB, gated).
dbutils.widgets.text("weight_models", "all", "Models to cache (comma-sep or 'all')")
_want = dbutils.widgets.get("weight_models").strip()
want = None if _want in ("", "all") else {s.strip() for s in _want.split(",")}

for key, m in MODELS.items():
    if m["weights_source"] != "hf":
        print(f"{key}: weights_source={m['weights_source']} (produced elsewhere, skipped)")
        continue
    if want is not None and key not in want:
        print(f"{key}: not in weight_models={_want} (skipped)")
        continue
    dest = f"{VOLUME_PATH}/{key}"
    if os.path.exists(f"{dest}/config.json"):
        print(f"{key}: cache HIT -> {dest}")
        continue
    try:
        print(f"{key}: downloading {m['hf_repo']} ...")
        tmp = snapshot_download(repo_id=m["hf_repo"], local_dir=tempfile.mkdtemp())
        os.makedirs(dest, exist_ok=True)
        shutil.copytree(tmp, dest, dirs_exist_ok=True)
        print(f"{key}: cached -> {dest}")
    except Exception as e:
        # Don't let one gated/missing model (e.g. Gemma without HF token) block the rest.
        print(f"{key}: DOWNLOAD FAILED (continuing) -- {type(e).__name__}: {str(e)[:200]}")

print("\nvolume contents:", os.listdir(VOLUME_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Done
# MAGIC Raw PII landed in the workspace catalog; base weights cached. Next: `01_lora_finetune_qwen`.

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"raw_rows": int(pii_pdf.shape[0]), "raw_table": RAW_TABLE}))
