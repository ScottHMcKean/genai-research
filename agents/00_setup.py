# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 00 · Setup — ensure the shared claims spine exists
# MAGIC
# MAGIC This demo builds on the shared claims data. If the tables aren't present, run
# MAGIC `fins_data/generate_data.py` first (it's idempotent — the "build from
# MAGIC scratch" step). This notebook just checks.

# COMMAND ----------

from config import CATALOG, SCHEMA, CLAIMS_TABLE, NOTES_TABLE, CHUNKS_TABLE, VOLUME_PATH

required = [CLAIMS_TABLE, NOTES_TABLE, CHUNKS_TABLE]
missing = [t for t in required if not spark.catalog.tableExists(t)]
if missing:
    raise RuntimeError(
        f"Missing tables {missing}. Run fins_data/generate_data.py first "
        f"(builds {CATALOG}.{SCHEMA} from scratch)."
    )

print("Claims spine present:")
for t in required:
    print(f"  {t}: {spark.table(t).count():,} rows")
print("Docs volume:", VOLUME_PATH)
display(dbutils.fs.ls(VOLUME_PATH))

# COMMAND ----------

# DBTITLE 1,Create claim_lookup UC function (Supervisor Agent tool)
from config import CATALOG, SCHEMA, CLAIMS_TABLE, CLAIM_LOOKUP_FN

# Idempotent — safe to re-run. Creates the UC function that the Supervisor
# Agent registers as a structured-data tool for claim ID lookups.
spark.sql(f"""
    CREATE FUNCTION IF NOT EXISTS {CLAIM_LOOKUP_FN}(p_claim_id STRING)
    RETURNS TABLE(
        claim_id     STRING,
        claim_type   STRING,
        peril        STRING,
        claim_status STRING,
        severity     STRING,
        claim_amount DOUBLE,
        region       STRING
    )
    COMMENT 'Look up a claim summary by claim_id. Used as a UC function tool by the Supervisor Agent.'
    RETURN
        SELECT claim_id, claim_type, peril, claim_status, severity, claim_amount, region
        FROM   {CLAIMS_TABLE}
        WHERE  claim_id = p_claim_id
""")

# Smoke-test: confirm one row comes back
result = spark.sql(f"SELECT * FROM {CLAIM_LOOKUP_FN}((SELECT claim_id FROM {CLAIMS_TABLE} LIMIT 1))").collect()
assert result, f"claim_lookup returned no rows — check {CLAIMS_TABLE} is populated"
print(f"OK  {CLAIM_LOOKUP_FN} → {result[0].asDict()}")
