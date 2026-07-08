# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · AI Governance — setup check
# MAGIC
# MAGIC The governance demo reads the **shared claims spine**. This notebook just verifies
# MAGIC it exists (idempotent) — it does **not** regenerate data.
# MAGIC
# MAGIC If anything is missing, run **`claims_demo/00_setup_and_data.py`** first (that is the
# MAGIC one "build from scratch" step for the whole suite).

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".")
from config import CATALOG, SCHEMA, CLAIMS_TABLE, NOTES_TABLE, CHUNKS_TABLE

required = [CLAIMS_TABLE, NOTES_TABLE, CHUNKS_TABLE]
missing = []
for t in required:
    try:
        n = spark.table(t).count()
        print(f"OK   {t:60s} rows={n}")
    except Exception as e:
        missing.append(t)
        print(f"MISS {t:60s} ({type(e).__name__})")

if missing:
    raise RuntimeError(
        "Shared claims data missing: "
        + ", ".join(missing)
        + "\n-> Run claims_demo/00_setup_and_data.py first."
    )
print("\nShared claims spine present. Governance demo is good to go.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this demo shows
# MAGIC | Notebook | Capability |
# MAGIC |---|---|
# MAGIC | `01_ai_gateway` | Unity **AI Gateway**: usage tracking, inference tables, rate limits, **PII guardrails** |
# MAGIC | `02_mcp_governance` | **MCP** on Databricks: managed MCP servers over UC functions / Vector Search / Genie, OBO auth, UC connections |
# MAGIC | `03_governance_observability` | System-table usage & **cost attribution**, **UC lineage** data → AI, payload logging |
