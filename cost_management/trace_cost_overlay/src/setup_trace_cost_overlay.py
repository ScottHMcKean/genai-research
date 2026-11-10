# Databricks notebook source
# MAGIC %md
# MAGIC # Corrected trace cost from authoritative Databricks cost data
# MAGIC
# MAGIC **Read-only overlay.** This notebook never writes to the MLflow-managed trace
# MAGIC tables. It reads traces and the Databricks system cost tables, and creates its
# MAGIC own views/tables in `output_catalog.output_schema`. Safe to drop.
# MAGIC
# MAGIC ## Why this replaces the old multiplier
# MAGIC MLflow trace cost (`mlflow.llm.cost`) is a per-token estimate from the LiteLLM
# MAGIC list table. It can use the wrong region/contract rate, and it ignores
# MAGIC provisioned-throughput economics. Scaling it by a $/DBU ratio mixed two bases
# MAGIC and made no sense. This version prices from the authoritative Databricks
# MAGIC sources instead:
# MAGIC
# MAGIC | Source | Role |
# MAGIC | --- | --- |
# MAGIC | `system.ai_gateway.external_model_spend` | external-model cost, already in USD |
# MAGIC | `system.ai_gateway.usage` | per-request tokens (`request_id`, cache detail) |
# MAGIC | `system.billing.usage` x `system.billing.list_prices` | Databricks-hosted DBU -> USD, region-correct, SCD2 by price time window |
# MAGIC | `system.serving.endpoint_usage` / `served_entities` | serving token detail |
# MAGIC
# MAGIC The region-correct price comes straight from `list_prices`, so the original
# MAGIC "0.07 vs 0.105" mismatch disappears with no rate config.
# MAGIC
# MAGIC ## Method
# MAGIC 1. Authoritative daily USD per model (`v_model_cost_daily`).
# MAGIC 2. Gateway billable tokens per model/day (`v_model_tokens_daily`).
# MAGIC 3. Effective rate = USD / tokens (`v_model_rate_daily`), with an SCD2 view.
# MAGIC 4. Price each request or span by its tokens, then sum per trace.
# MAGIC
# MAGIC Allocation by token share makes the total reconcile to billing by construction.
# MAGIC A model/day with billing but no gateway traffic shows as an unattributed gap
# MAGIC (non-gateway usage). For provisioned endpoints the per-token rate is high
# MAGIC (reserved capacity, not a token price); `billing_basis` flags this.

# COMMAND ----------

dbutils.widgets.text("trace_catalog", "uaigw_fe", "Trace catalog")
dbutils.widgets.text("trace_schema", "uaigw_schema", "Trace schema")
dbutils.widgets.text("table_prefix", "unity_gateway", "Trace table prefix")
dbutils.widgets.text("output_catalog", "uaigw_fe", "Output catalog")
dbutils.widgets.text("output_schema", "trace_cost_overlay", "Output schema")
dbutils.widgets.dropdown("trace_mode", "gateway", ["gateway", "mlflow_span"], "Trace link mode")

TRACE_CATALOG = dbutils.widgets.get("trace_catalog")
TRACE_SCHEMA = dbutils.widgets.get("trace_schema")
TABLE_PREFIX = dbutils.widgets.get("table_prefix")
OUT = f'{dbutils.widgets.get("output_catalog")}.{dbutils.widgets.get("output_schema")}'
TRACE_MODE = dbutils.widgets.get("trace_mode")
SPANS = f"{TRACE_CATALOG}.{TRACE_SCHEMA}.{TABLE_PREFIX}_otel_spans"
print(f"Spans (read-only): {SPANS}\nOutput: {OUT}\nTrace mode: {TRACE_MODE}")

# trace_mode:
#   gateway     - AI Gateway unified tracing. Link spans to requests by databricks.request_id.
#   mlflow_span - MLflow app tracing. Reprice each LLM span's own tokens by the model rate.

# COMMAND ----------

# MAGIC %md ## Step 1 - Discover (inspect before building)

# COMMAND ----------

spark.sql(f"DESCRIBE {SPANS}").show(50, truncate=False)
spark.sql(f"""
  SELECT count(*) spans,
         count(attributes:['databricks.request_id']) with_request_id,
         count(attributes:['mlflow.llm.cost'])        with_mlflow_cost,
         count(attributes:['mlflow.chat.tokenUsage'])  with_tokens,
         min(date) earliest, max(date) latest
  FROM {SPANS}
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Step 2 - Config: contract overrides (optional; default = list price)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUT} COMMENT 'Interim read-only corrected-trace-cost overlay. Safe to drop.'")
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {OUT}.trace_cost_rate_contract (
    model_pattern STRING, region STRING, discount_factor DOUBLE,
    valid_from DATE, valid_to DATE, priority INT, notes STRING
  ) COMMENT 'Negotiated discount off list for Databricks-hosted models. 1.0 = list. External already bills actual USD.'
""")
spark.sql(f"""
  INSERT OVERWRITE {OUT}.trace_cost_rate_contract VALUES
  ('%', '', 1.0, DATE'2026-01-01', NULL, 100,
   'Default: list price (region-correct via list_prices). Add a customer row, e.g. (''claude%'',''northeurope'',0.88,...,10) for 12% off.')
""")

# COMMAND ----------

# MAGIC %md ## Step 3 - Authoritative daily USD per model

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_model_cost_daily AS
WITH dbx AS (
  SELECT u.usage_date,
    regexp_replace(u.usage_metadata.endpoint_name, '^(system\\\\.ai\\\\.)?(databricks-)?', '') AS model_key,
    'databricks' AS cost_source,
    sum(u.usage_quantity * try_cast(lp.pricing.default AS double)) AS usd_list
  FROM system.billing.usage u
  JOIN system.billing.list_prices lp
    ON u.sku_name = lp.sku_name
   AND u.usage_end_time >= lp.price_start_time
   AND (lp.price_end_time IS NULL OR u.usage_end_time < lp.price_end_time)
  WHERE u.usage_unit = 'DBU' AND u.sku_name LIKE 'ENTERPRISE_%_MODEL_SERVING'
    AND u.usage_metadata.endpoint_name IS NOT NULL
  GROUP BY 1, 2, 3
),
ext AS (
  SELECT usage_date,
    regexp_replace(usage_metadata.model, '^(system\\\\.ai\\\\.)?(databricks-)?', '') AS model_key,
    'external' AS cost_source,
    sum(try_cast(usage_quantity AS double)) AS usd_list
  FROM system.ai_gateway.external_model_spend WHERE usage_unit = 'USD' GROUP BY 1, 2, 3
),
unioned AS (SELECT * FROM dbx UNION ALL SELECT * FROM ext)
SELECT u.usage_date, u.model_key, u.cost_source,
  CASE WHEN u.cost_source = 'databricks' THEN u.usd_list * coalesce(ct.discount_factor, 1) ELSE u.usd_list END AS usd,
  u.usd_list,
  CASE WHEN u.cost_source = 'databricks' THEN coalesce(ct.discount_factor, 1) ELSE 1 END AS discount_factor
FROM unioned u
LEFT JOIN {OUT}.trace_cost_rate_contract ct
  ON u.cost_source = 'databricks' AND u.model_key LIKE ct.model_pattern
 AND u.usage_date BETWEEN ct.valid_from AND coalesce(ct.valid_to, DATE'9999-12-31')
QUALIFY row_number() OVER (PARTITION BY u.usage_date, u.model_key, u.cost_source ORDER BY ct.priority ASC NULLS LAST) = 1
""")

# COMMAND ----------

# MAGIC %md ## Step 4 - Gateway billable tokens and effective rate (with SCD2 view)

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_model_tokens_daily AS
SELECT to_date(event_time) AS usage_date,
  regexp_replace(destination_model, '^(system\\\\.ai\\\\.)?(databricks-)?', '') AS model_key,
  CASE WHEN destination_type = 'EXTERNAL_FOUNDATION_MODEL' THEN 'external' ELSE 'databricks' END AS cost_source,
  sum(CASE WHEN destination_type = 'EXTERNAL_FOUNDATION_MODEL'
           THEN coalesce(input_tokens,0) + coalesce(output_tokens,0)
           ELSE greatest(coalesce(input_tokens,0) - coalesce(token_details.cache_read_input_tokens,0), 0) + coalesce(output_tokens,0)
      END) AS tokens
FROM system.ai_gateway.usage WHERE destination_type <> '' GROUP BY 1, 2, 3
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_model_rate_daily AS
SELECT c.usage_date, c.model_key, c.cost_source, c.usd, c.usd_list, c.discount_factor,
  t.tokens, c.usd / nullif(t.tokens, 0) AS usd_per_token,
  CASE WHEN c.usd / nullif(t.tokens, 0) * 1e6 < 200 THEN 'pay_per_token' ELSE 'provisioned_or_sparse' END AS billing_basis
FROM {OUT}.v_model_cost_daily c
JOIN {OUT}.v_model_tokens_daily t USING (usage_date, model_key, cost_source)
""")

# Price history (demonstrates list_prices SCD2) and collapsed rate periods.
spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_list_price_history AS
SELECT sku_name, try_cast(pricing.default AS double) AS usd_per_dbu, price_start_time, price_end_time
FROM system.billing.list_prices WHERE sku_name LIKE 'ENTERPRISE_%_MODEL_SERVING'
""")
spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_model_rate_scd2 AS
WITH r AS (SELECT model_key, cost_source, usage_date, round(usd_per_token, 10) AS rate FROM {OUT}.v_model_rate_daily),
flagged AS (SELECT *, CASE WHEN rate = lag(rate) OVER (PARTITION BY model_key, cost_source ORDER BY usage_date) THEN 0 ELSE 1 END AS chg FROM r),
grp AS (SELECT *, sum(chg) OVER (PARTITION BY model_key, cost_source ORDER BY usage_date) AS g FROM flagged)
SELECT model_key, cost_source, rate AS usd_per_token, min(usage_date) AS valid_from, max(usage_date) AS valid_to, count(*) AS days
FROM grp GROUP BY model_key, cost_source, rate, g
""")

# COMMAND ----------

# MAGIC %md ## Step 5 - Per-request cost (gateway) and per-trace cost

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_request_cost AS
WITH u AS (
  SELECT request_id, event_time, to_date(event_time) AS usage_date,
    endpoint_name AS gateway_endpoint, destination_type,
    regexp_replace(destination_model, '^(system\\\\.ai\\\\.)?(databricks-)?', '') AS model_key,
    CASE WHEN destination_type = 'EXTERNAL_FOUNDATION_MODEL' THEN 'external' ELSE 'databricks' END AS cost_source,
    coalesce(input_tokens,0) AS input_tokens, coalesce(output_tokens,0) AS output_tokens,
    coalesce(token_details.cache_read_input_tokens,0) AS cache_read_tokens,
    coalesce(token_details.cache_creation_input_tokens,0) AS cache_creation_tokens,
    request_tags, status_code
  FROM system.ai_gateway.usage WHERE destination_type <> ''
)
SELECT u.*,
  CASE WHEN u.cost_source = 'databricks'
       THEN greatest(u.input_tokens - u.cache_read_tokens, 0) + u.output_tokens
       ELSE u.input_tokens + u.output_tokens END AS priced_tokens,
  r.usd_per_token, r.billing_basis,
  (CASE WHEN u.cost_source = 'databricks'
        THEN greatest(u.input_tokens - u.cache_read_tokens, 0) + u.output_tokens
        ELSE u.input_tokens + u.output_tokens END) * r.usd_per_token AS request_cost,
  (r.usd_per_token IS NOT NULL) AS rate_matched
FROM u
LEFT JOIN {OUT}.v_model_rate_daily r
  ON u.usage_date = r.usage_date AND u.model_key = r.model_key AND u.cost_source = r.cost_source
""")

# COMMAND ----------

if TRACE_MODE == "gateway":
    # Link spans to gateway requests by databricks.request_id; sum request cost per trace.
    spark.sql(f"""
    CREATE OR REPLACE VIEW {OUT}.v_trace_cost_corrected AS
    WITH span_req AS (
      SELECT DISTINCT trace_id, nullif(attributes:['databricks.request_id']::string, '') AS request_id
      FROM {SPANS} WHERE attributes:['databricks.request_id'] IS NOT NULL
    )
    SELECT sr.trace_id, min(rc.usage_date) AS trace_date, min(rc.event_time) AS trace_start_time,
      count(*) AS request_count,
      sum(rc.input_tokens) AS input_tokens, sum(rc.output_tokens) AS output_tokens,
      sum(rc.cache_read_tokens) AS cache_read_tokens,
      sum(rc.request_cost) AS corrected_cost,
      array_join(sort_array(collect_set(rc.model_key)), ',') AS models,
      array_join(sort_array(collect_set(rc.cost_source)), ',') AS cost_sources,
      array_join(sort_array(collect_set(rc.billing_basis)), ',') AS billing_bases,
      (count_if(NOT rc.rate_matched) = 0) AS all_rates_matched
    FROM span_req sr JOIN {OUT}.v_request_cost rc USING (request_id)
    GROUP BY sr.trace_id
    """)
else:
    # MLflow app tracing: reprice each LLM span's own tokens by the model rate.
    spark.sql(f"""
    CREATE OR REPLACE VIEW {OUT}.v_trace_cost_corrected AS
    WITH spans AS (
      SELECT trace_id, date AS span_date,
        regexp_replace(coalesce(nullif(attributes:['gen_ai.request.model']::string,''),
                                nullif(attributes:['mlflow.llm.model']::string,'')),
                       '^(system\\\\.ai\\\\.)?(databricks-)?', '') AS model_key,
        attributes:['mlflow.llm.cost']['total_cost']::double AS mlflow_cost,
        coalesce(attributes:['mlflow.chat.tokenUsage']['input_tokens']::bigint, 0)  AS input_tokens,
        coalesce(attributes:['mlflow.chat.tokenUsage']['output_tokens']::bigint, 0) AS output_tokens,
        coalesce(attributes:['mlflow.chat.tokenUsage']['cache_read_input_tokens']::bigint, 0) AS cache_read_tokens
      FROM {SPANS} WHERE attributes:['mlflow.llm.cost'] IS NOT NULL
    )
    SELECT s.trace_id, min(s.span_date) AS trace_date,
      sum(s.input_tokens) AS input_tokens, sum(s.output_tokens) AS output_tokens,
      sum(s.cache_read_tokens) AS cache_read_tokens,
      round(sum(s.mlflow_cost), 6) AS mlflow_total_cost,
      sum((greatest(s.input_tokens - s.cache_read_tokens, 0) + s.output_tokens) * r.usd_per_token) AS corrected_cost,
      array_join(sort_array(collect_set(s.model_key)), ',') AS models,
      array_join(sort_array(collect_set(r.billing_basis)), ',') AS billing_bases,
      (count_if(r.usd_per_token IS NULL) = 0) AS all_rates_matched
    FROM spans s
    LEFT JOIN {OUT}.v_model_rate_daily r
      ON s.span_date = r.usage_date AND s.model_key = r.model_key AND r.cost_source = 'databricks'
    GROUP BY s.trace_id
    """)
print(f"Created v_trace_cost_corrected ({TRACE_MODE} mode)")

# COMMAND ----------

# MAGIC %md ## Step 6 - Reconciliation vs authoritative USD

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_trace_cost_reconciliation AS
WITH attributed AS (
  SELECT usage_date, cost_source, sum(request_cost) AS attributed_usd
  FROM {OUT}.v_request_cost WHERE rate_matched GROUP BY 1, 2
),
authoritative AS (
  SELECT usage_date, cost_source, sum(usd) AS authoritative_usd
  FROM {OUT}.v_model_cost_daily GROUP BY 1, 2
)
SELECT coalesce(b.usage_date, a.usage_date) AS usage_date,
  coalesce(b.cost_source, a.cost_source) AS cost_source,
  round(b.authoritative_usd, 2) AS authoritative_usd,
  round(a.attributed_usd, 2) AS attributed_usd,
  round(b.authoritative_usd - a.attributed_usd, 2) AS unattributed_usd,
  round(100 * a.attributed_usd / nullif(b.authoritative_usd, 0), 1) AS attributed_pct
FROM authoritative b FULL OUTER JOIN attributed a USING (usage_date, cost_source)
""")

# COMMAND ----------

# MAGIC %md ## Validation

# COMMAND ----------

spark.sql(f"""
  SELECT round(sum(usd_per_token * tokens), 2) AS rate_times_tokens,
         round(sum(usd), 2) AS authoritative_usd
  FROM {OUT}.v_model_rate_daily WHERE usage_date >= current_date()-INTERVAL 7 DAYS AND cost_source='databricks'
""").show(truncate=False)
spark.sql(f"""
  SELECT cost_source, round(avg(attributed_pct),1) AS avg_attributed_pct
  FROM {OUT}.v_trace_cost_reconciliation WHERE usage_date >= current_date()-INTERVAL 7 DAYS GROUP BY 1
""").show(truncate=False)
spark.sql(f"SELECT count(*) traces, round(sum(corrected_cost),2) usd FROM {OUT}.v_trace_cost_corrected").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Teardown (remove the overlay when the product fix lands)

# COMMAND ----------

# for v in ['v_trace_cost_reconciliation','v_trace_cost_corrected','v_request_cost','v_model_rate_scd2',
#           'v_list_price_history','v_model_rate_daily','v_model_tokens_daily','v_model_cost_daily']:
#     spark.sql(f"DROP VIEW IF EXISTS {OUT}.{v}")
# spark.sql(f"DROP TABLE IF EXISTS {OUT}.trace_cost_corrected")
# spark.sql(f"DROP TABLE IF EXISTS {OUT}.trace_cost_rate_contract")
# spark.sql(f"DROP SCHEMA IF EXISTS {OUT}")
