# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Observability & cost attribution for AI
# MAGIC
# MAGIC Governance isn't just access control — it's *knowing what happened*. Because every
# MAGIC model call goes through serving + the gateway, you get **system-table** telemetry,
# MAGIC **cost attribution**, **UC lineage** (data → model → agent), and **payload logs** —
# MAGIC all in Unity Catalog, queryable with SQL.
# MAGIC
# MAGIC System tables can lag a few hours and some require enablement, so every query below is
# MAGIC defensive: it prints guidance instead of failing.

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".")
from config import CATALOG, SCHEMA, GATEWAY_ENDPOINT

def try_sql(title, sql):
    print(f"\n### {title}")
    try:
        df = spark.sql(sql)
        rows = df.limit(20).toPandas()
        if len(rows) == 0:
            print("(query ran, no rows yet — telemetry may still be populating)")
        else:
            print(rows.to_string(index=False))
        return df
    except Exception as e:
        print(f"(skipped: {type(e).__name__}: {str(e)[:180]})")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Serving usage & token spend (system.serving)
# MAGIC Per-endpoint request counts, tokens and latency — the raw material for chargeback.

# COMMAND ----------

try_sql(
    "Requests & tokens by endpoint (last 7 days)",
    """
    SELECT served_entity_name,
           COUNT(*)                         AS requests,
           SUM(COALESCE(input_token_count,0))  AS input_tokens,
           SUM(COALESCE(output_token_count,0)) AS output_tokens
    FROM system.serving.endpoint_usage
    WHERE request_time >= current_timestamp() - INTERVAL 7 DAYS
    GROUP BY served_entity_name
    ORDER BY requests DESC
    """,
)
# Schema of system.serving tables varies by release; if the above errors, inspect columns:
try_sql("Available system.serving tables", "SHOW TABLES IN system.serving")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Cost attribution (system.billing.usage)
# MAGIC Model-serving DBUs by day — attribute AI spend to teams/endpoints.

# COMMAND ----------

try_sql(
    "Model-serving DBUs by day (last 14 days)",
    """
    SELECT usage_date,
           sku_name,
           ROUND(SUM(usage_quantity), 2) AS dbus
    FROM system.billing.usage
    WHERE usage_date >= current_date() - INTERVAL 14 DAYS
      AND (lower(sku_name) LIKE '%serving%' OR lower(sku_name) LIKE '%inference%'
           OR lower(sku_name) LIKE '%gpu%' OR lower(sku_name) LIKE '%ai%')
    GROUP BY usage_date, sku_name
    ORDER BY usage_date DESC, dbus DESC
    """,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Gateway payload / inference log
# MAGIC The audit table the gateway writes (from notebook 01). Every request+response, governed.

# COMMAND ----------

payload = f"{CATALOG}.{SCHEMA}.gateway_claims_llm_payload"
df = try_sql(f"Inference log: {payload}", f"SELECT * FROM {payload} ORDER BY 1 DESC LIMIT 5")
if df is None:
    print("\n-> Run 01_ai_gateway first and send a few requests; logging lags a few minutes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Lineage: claims data → AI
# MAGIC `system.access.table_lineage` ties the claims table to everything downstream (features,
# MAGIC functions, models). This is how you answer "what fed this model?" for audit/model risk.

# COMMAND ----------

try_sql(
    "Downstream lineage from the claims table",
    f"""
    SELECT source_table_full_name, target_table_full_name, entity_type, event_time
    FROM system.access.table_lineage
    WHERE source_table_full_name = '{CATALOG}.{SCHEMA}.claims'
    ORDER BY event_time DESC
    LIMIT 20
    """,
)
print("\nUI: open the `claims` table > Lineage tab for the visual graph "
      "(data -> functions -> vector index -> agents).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Talk track
# MAGIC - **Every AI call is telemetry.** Requests, tokens, latency, and cost land in governed
# MAGIC   system tables — no agents, no scraping.
# MAGIC - **Chargeback-ready.** Attribute spend to endpoint / team / user from `system.billing`.
# MAGIC - **Model-risk answers on demand.** UC lineage shows exactly which data fed which model
# MAGIC   or agent — the question model-validation teams always ask.
