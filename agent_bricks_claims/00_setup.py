# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup — ensure the shared claims spine exists
# MAGIC
# MAGIC This demo builds on the shared claims data. If the tables aren't present, run
# MAGIC `claims_demo/00_setup_and_data.py` first (it's idempotent — the "build from
# MAGIC scratch" step). This notebook just checks.

# COMMAND ----------

from config import CATALOG, SCHEMA, CLAIMS_TABLE, NOTES_TABLE, CHUNKS_TABLE, VOLUME_PATH

required = [CLAIMS_TABLE, NOTES_TABLE, CHUNKS_TABLE]
missing = [t for t in required if not spark.catalog.tableExists(t)]
if missing:
    raise RuntimeError(
        f"Missing tables {missing}. Run claims_demo/00_setup_and_data.py first "
        f"(builds {CATALOG}.{SCHEMA} from scratch)."
    )

print("Claims spine present:")
for t in required:
    print(f"  {t}: {spark.table(t).count():,} rows")
print("Docs volume:", VOLUME_PATH)
display(dbutils.fs.ls(VOLUME_PATH))
