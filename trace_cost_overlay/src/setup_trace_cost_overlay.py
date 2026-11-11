# Databricks notebook source
# MAGIC %md
# MAGIC # Interim corrected-trace-cost overlay — setup, views, validation, reconciliation
# MAGIC
# MAGIC **Read-only overlay.** This notebook never writes to the MLflow-managed trace
# MAGIC tables/views (`<prefix>_otel_spans|_otel_logs|_otel_metrics|_otel_annotations`,
# MAGIC `<prefix>_trace_metadata`, `<prefix>_trace_unified`). It only reads them and
# MAGIC creates its own objects in `output_catalog.output_schema`. Safe to drop
# MAGIC (see the teardown cell at the end) when the product fix lands.
# MAGIC
# MAGIC **Why this exists.** MLflow trace cost is estimated at an AWS/GCP list rate
# MAGIC (~$0.07/DBU) instead of the customer's Azure North Europe rate ($0.105/DBU
# MAGIC list, ~$0.092/DBU effective). This overlay recomputes a corrected cost.
# MAGIC
# MAGIC **Two pricing engines**, chosen per span from the attributes present:
# MAGIC 1. `dbu_multiplier` — span carries `mlflow.llm.cost`. corrected = mlflow_total_cost
# MAGIC    × (target_rate / mlflow_assumed_rate). This is the customer's data shape.
# MAGIC 2. `token` — span carries only token usage (`mlflow.chat.tokenUsage` /
# MAGIC    `gen_ai.usage.*`), no cost. corrected = tokens × per-token rate. This is the
# MAGIC    AI-Gateway unified-tracing shape (e.g. the uaigw test bed), and it is also
# MAGIC    the spec's fallback when mlflow cost does not imply a flat $/DBU.

# COMMAND ----------

dbutils.widgets.text("trace_catalog", "uaigw_fe", "Trace catalog")
dbutils.widgets.text("trace_schema", "uaigw_schema", "Trace schema")
dbutils.widgets.text("table_prefix", "unity_gateway", "Trace table prefix")
dbutils.widgets.text("output_catalog", "uaigw_fe", "Output catalog")
dbutils.widgets.text("output_schema", "trace_cost_overlay", "Output schema")
dbutils.widgets.dropdown("pricing_mode", "effective", ["effective", "azure_list"], "Pricing mode")

TRACE_CATALOG = dbutils.widgets.get("trace_catalog")
TRACE_SCHEMA = dbutils.widgets.get("trace_schema")
TABLE_PREFIX = dbutils.widgets.get("table_prefix")
OUT_CATALOG = dbutils.widgets.get("output_catalog")
OUT_SCHEMA = dbutils.widgets.get("output_schema")
PRICING_MODE = dbutils.widgets.get("pricing_mode")

SPANS = f"{TRACE_CATALOG}.{TRACE_SCHEMA}.{TABLE_PREFIX}_otel_spans"
OUT = f"{OUT_CATALOG}.{OUT_SCHEMA}"
print(f"Spans (read-only): {SPANS}")
print(f"Overlay output:    {OUT}")
print(f"Pricing mode:      {PRICING_MODE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Discover. Inspect the spans schema and sample rows before any SQL.
# MAGIC Confirm the attribute keys from data; do not assume them.

# COMMAND ----------

print("=== columns ===")
spark.sql(f"DESCRIBE {SPANS}").show(50, truncate=False)

# COMMAND ----------

# Which attribute keys actually exist on cost/token-bearing spans?
spark.sql(f"""
  SELECT DISTINCT k FROM (
    SELECT explode(map_keys(try_cast(attributes AS MAP<STRING,STRING>))) AS k
    FROM {SPANS}
    WHERE attributes:['mlflow.chat.tokenUsage'] IS NOT NULL
       OR attributes:['mlflow.llm.cost'] IS NOT NULL
    LIMIT 500
  ) ORDER BY k
""").show(100, truncate=False)

# COMMAND ----------

# Presence counts. If spans_with_cost = 0 the workspace uses token-only tracing
# (AI Gateway shape) and the token engine is the only applicable path.
spark.sql(f"""
  SELECT
    count(*)                                            AS total_spans,
    count(attributes:['mlflow.llm.cost'])               AS spans_with_cost,
    count(attributes:['mlflow.chat.tokenUsage'])        AS spans_with_tokens,
    count_if(attributes:['mlflow.chat.tokenUsage'] IS NOT NULL
             AND attributes:['mlflow.llm.cost'] IS NULL) AS tokens_no_cost,
    min(date) AS earliest_date, max(date) AS latest_date
  FROM {SPANS}
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Config: settings + rate config table (rates never hardcoded)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUT} COMMENT 'Interim read-only corrected-trace-cost overlay. Safe to drop.'")

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {OUT}.trace_cost_settings (
    setting_key STRING, setting_value STRING
  ) COMMENT 'Runtime settings (e.g. pricing_mode) read by the views.'
""")
spark.sql(f"INSERT OVERWRITE {OUT}.trace_cost_settings VALUES ('pricing_mode', '{PRICING_MODE}')")

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {OUT}.trace_cost_rate_config (
    model_pattern STRING,
    cloud STRING,
    region STRING,
    pricing_method STRING,          -- 'token' | 'dbu_multiplier'
    input_rate_per_mtok DOUBLE,     -- token method: USD per 1M input tokens
    output_rate_per_mtok DOUBLE,    -- token method: USD per 1M output tokens
    cache_read_rate_per_mtok DOUBLE,-- token method: USD per 1M cached-read input tokens
    mlflow_assumed_rate DOUBLE,     -- dbu method: $/DBU MLflow assumed (e.g. 0.07)
    azure_list_rate DOUBLE,         -- dbu method: $/DBU Azure list (0.105)
    effective_rate DOUBLE,          -- dbu method: $/DBU effective (0.092)
    valid_from DATE,
    valid_to DATE,
    priority INT,                   -- lower wins on overlapping patterns; >=100 = default (rate_matched=false)
    notes STRING
  ) COMMENT 'Rate config for the interim trace-cost overlay. Read-only overlay; safe to drop.'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seed rates
# MAGIC `valid_from` = earliest trace date. The `claude% / AZURE / northeurope /
# MAGIC dbu_multiplier` row is the **customer scenario** (0.07 assumed, 0.105 list,
# MAGIC 0.092 effective). The `token` rows are **PLACEHOLDERS** for the test bed —
# MAGIC replace per-Mtok values with real rates before trusting dollar figures.
# MAGIC To change a rate: update/insert a row here (or edit the table) and re-run the
# MAGIC views — no code change needed.

# COMMAND ----------

earliest = spark.sql(f"SELECT coalesce(min(date), current_date()) FROM {SPANS}").collect()[0][0]
vf = earliest.strftime("%Y-%m-%d")
print(f"valid_from = {vf}")

spark.sql(f"""
  INSERT OVERWRITE {OUT}.trace_cost_rate_config VALUES
  ('claude%','AZURE','northeurope','dbu_multiplier', NULL,NULL,NULL, 0.07,0.105,0.092, DATE'{vf}',NULL, 10, 'Customer scenario: MLflow assumed 0.07/DBU; Azure NE list 0.105, effective 0.092. Applies only to spans carrying mlflow.llm.cost.'),
  ('claude-opus%','AZURE','northeurope','token', 15,75,1.5, NULL,NULL,NULL, DATE'{vf}',NULL, 20, 'PLACEHOLDER per-Mtok rates for test bed.'),
  ('claude-sonnet%','AZURE','northeurope','token', 3,15,0.30, NULL,NULL,NULL, DATE'{vf}',NULL, 20, 'PLACEHOLDER.'),
  ('claude-haiku%','AZURE','northeurope','token', 0.80,4,0.08, NULL,NULL,NULL, DATE'{vf}',NULL, 20, 'PLACEHOLDER.'),
  ('claude%','AZURE','northeurope','token', 3,15,0.30, NULL,NULL,NULL, DATE'{vf}',NULL, 40, 'PLACEHOLDER catch-all Claude.'),
  ('gpt-6%','AZURE','northeurope','token', 2,16,0.20, NULL,NULL,NULL, DATE'{vf}',NULL, 20, 'PLACEHOLDER.'),
  ('gpt-5-nano%','AZURE','northeurope','token', 0.05,0.40,0.005, NULL,NULL,NULL, DATE'{vf}',NULL, 20, 'PLACEHOLDER.'),
  ('gpt-5%','AZURE','northeurope','token', 1.25,10,0.125, NULL,NULL,NULL, DATE'{vf}',NULL, 30, 'PLACEHOLDER.'),
  ('gpt%','AZURE','northeurope','token', 1.25,10,0.125, NULL,NULL,NULL, DATE'{vf}',NULL, 40, 'PLACEHOLDER catch-all GPT.'),
  ('kimi%','AZURE','northeurope','token', 0.60,2.5,0.06, NULL,NULL,NULL, DATE'{vf}',NULL, 30, 'PLACEHOLDER.'),
  ('glm%','AZURE','northeurope','token', 0.60,2.2,0.06, NULL,NULL,NULL, DATE'{vf}',NULL, 30, 'PLACEHOLDER.'),
  ('%','AZURE','northeurope','token', 2,8,0.20, NULL,NULL,NULL, DATE'{vf}',NULL, 100, 'PLACEHOLDER default; matches any model. rate_matched=false upstream.')
""")
spark.sql(f"SELECT * FROM {OUT}.trace_cost_rate_config ORDER BY priority, model_pattern").show(50, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — View: one row per cost-bearing (LLM) span, with corrected cost
# MAGIC Unmatched models are **passed through, never dropped**, and flagged
# MAGIC `rate_matched = false`. `is_topmost_cost_span` marks the span to keep after
# MAGIC dedup (see Step 4).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_trace_span_cost_corrected AS
WITH settings AS (
  SELECT max(CASE WHEN setting_key = 'pricing_mode' THEN setting_value END) AS pricing_mode
  FROM {OUT}.trace_cost_settings
),
raw AS (
  SELECT
    trace_id, span_id, parent_span_id,
    date AS span_date, time AS span_time,
    attributes:['mlflow.spanType']::string       AS span_type,
    attributes:['gen_ai.operation.name']::string AS operation,
    nullif(attributes:['databricks.endpoint_name']::string, '') AS endpoint_name,
    coalesce(nullif(attributes:['gen_ai.request.model']::string, ''),
             nullif(attributes:['mlflow.llm.model']::string, ''))  AS model_attr,
    nullif(attributes:['gen_ai.provider.name']::string, '')      AS provider_attr,
    attributes:['mlflow.llm.cost']['total_cost']::double  AS mlflow_total_cost,
    parse_json(attributes:['mlflow.chat.tokenUsage']::string) AS tu,
    (attributes:['mlflow.llm.cost'] IS NOT NULL)        AS has_cost_attr,
    (attributes:['mlflow.chat.tokenUsage'] IS NOT NULL) AS has_tok_attr
  FROM {SPANS}
),
base AS (
  SELECT *,
    coalesce(tu:input_tokens::bigint, 0)            AS input_tokens,
    coalesce(tu:output_tokens::bigint, 0)           AS output_tokens,
    coalesce(tu:cache_read_input_tokens::bigint, 0) AS cache_read_tokens,
    (has_cost_attr OR has_tok_attr)                 AS is_cost_bearing
  FROM raw
),
child_model AS (
  SELECT trace_id, parent_span_id,
         max(model_attr)    AS child_model,
         max(provider_attr) AS child_provider
  FROM base
  WHERE is_cost_bearing AND model_attr IS NOT NULL
  GROUP BY trace_id, parent_span_id
),
enriched AS (
  SELECT b.*,
    (p.span_id IS NULL OR NOT p.is_cost_bearing) AS is_topmost_cost_span,
    coalesce(
      regexp_replace(
        coalesce(b.model_attr, cm.child_model, b.endpoint_name),
        '^(system\\\\.ai\\\\.)?(databricks-)?', ''),
      '(unknown)'
    ) AS model_key,
    coalesce(b.provider_attr, cm.child_provider) AS provider,
    CASE WHEN b.mlflow_total_cost IS NOT NULL THEN 'dbu_multiplier' ELSE 'token' END AS pricing_method
  FROM base b
  LEFT JOIN base p         ON b.trace_id = p.trace_id AND b.parent_span_id = p.span_id
  LEFT JOIN child_model cm ON b.trace_id = cm.trace_id AND b.span_id = cm.parent_span_id
  WHERE b.is_cost_bearing
),
matched AS (
  SELECT e.*,
    c.model_pattern, c.cloud AS rate_cloud, c.region AS rate_region, c.priority AS rate_priority,
    c.input_rate_per_mtok, c.output_rate_per_mtok, c.cache_read_rate_per_mtok,
    c.mlflow_assumed_rate, c.azure_list_rate, c.effective_rate,
    row_number() OVER (PARTITION BY e.trace_id, e.span_id ORDER BY c.priority ASC NULLS LAST) AS rn
  FROM enriched e
  LEFT JOIN {OUT}.trace_cost_rate_config c
    ON e.model_key LIKE c.model_pattern
   AND c.pricing_method = e.pricing_method
   AND e.span_date BETWEEN c.valid_from AND coalesce(c.valid_to, DATE'9999-12-31')
),
priced AS (
  -- Resolve the target rate and multiplier once (was computed twice).
  SELECT m.*, s.pricing_mode,
    CASE WHEN m.pricing_method = 'dbu_multiplier'
      THEN (CASE s.pricing_mode WHEN 'azure_list' THEN m.azure_list_rate ELSE m.effective_rate END)
           / nullif(m.mlflow_assumed_rate, 0)
    END AS multiplier
  FROM matched m
  CROSS JOIN settings s
  WHERE m.rn = 1
)
SELECT
  trace_id, span_id, parent_span_id, span_date, span_time,
  span_type, operation, endpoint_name, model_key, provider,
  is_topmost_cost_span, pricing_method, pricing_mode,
  input_tokens, output_tokens, cache_read_tokens,
  greatest(input_tokens - cache_read_tokens, 0) AS billable_input_tokens,
  mlflow_total_cost, multiplier,
  CASE
    -- dbu: multiply; unmatched (multiplier NULL) passes through at 1x.
    WHEN pricing_method = 'dbu_multiplier'
      THEN mlflow_total_cost * coalesce(multiplier, 1)
    ELSE
      greatest(input_tokens - cache_read_tokens, 0) / 1e6 * input_rate_per_mtok
      + cache_read_tokens / 1e6 * cache_read_rate_per_mtok
      + output_tokens     / 1e6 * output_rate_per_mtok
  END AS corrected_cost,
  (model_pattern IS NOT NULL AND coalesce(rate_priority, 999) < 100) AS rate_matched,
  model_pattern AS rate_pattern_used, rate_cloud, rate_region, rate_priority,
  input_rate_per_mtok, output_rate_per_mtok, cache_read_rate_per_mtok,
  mlflow_assumed_rate, azure_list_rate, effective_rate
FROM priced
""")
print("Created view v_trace_span_cost_corrected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — View: one row per trace (dedup nested spans)
# MAGIC MLflow dedupes nested spans: a span is skipped if an ancestor already carries
# MAGIC the same cost/token attribute. We reproduce that by keeping only the
# MAGIC **top-most** cost-bearing span in each branch (`is_topmost_cost_span`).

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_trace_cost_corrected AS
SELECT
  trace_id,
  min(span_date) AS trace_date,
  min(span_time) AS trace_start_time,
  max(span_time) AS trace_end_time,
  count(*)       AS llm_span_count,
  sum(input_tokens)          AS input_tokens,
  sum(cache_read_tokens)     AS cache_read_tokens,
  sum(billable_input_tokens) AS billable_input_tokens,
  sum(output_tokens)         AS output_tokens,
  sum(mlflow_total_cost)     AS mlflow_total_cost,
  sum(corrected_cost)        AS corrected_cost,
  max(pricing_mode)          AS pricing_mode,
  (count_if(NOT rate_matched) = 0) AS all_rates_matched,
  count_if(NOT rate_matched)       AS unmatched_span_count,
  array_join(sort_array(collect_set(model_key)), ',')      AS models,
  array_join(sort_array(collect_set(pricing_method)), ',') AS pricing_methods
FROM {OUT}.v_trace_span_cost_corrected
WHERE is_topmost_cost_span = true
GROUP BY trace_id
""")
print("Created view v_trace_cost_corrected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4b — Validation
# MAGIC 1. Dedup check: top-most span sum == trace-view sum, and < naive all-span sum.
# MAGIC 2. mlflow.trace.cost match: only possible when a `<prefix>_trace_metadata`
# MAGIC    table with `mlflow.trace.cost` exists. On token-only (AI Gateway) workspaces
# MAGIC    there is no trace-level cost, so this check is reported as N/A.

# COMMAND ----------

spark.sql(f"""
  SELECT
    (SELECT round(sum(corrected_cost),2) FROM {OUT}.v_trace_span_cost_corrected
       WHERE span_date >= current_date()-INTERVAL 30 DAYS) AS naive_all_span_cost,
    (SELECT round(sum(corrected_cost),2) FROM {OUT}.v_trace_span_cost_corrected
       WHERE is_topmost_cost_span AND span_date >= current_date()-INTERVAL 30 DAYS) AS topmost_span_cost,
    (SELECT round(sum(corrected_cost),2) FROM {OUT}.v_trace_cost_corrected
       WHERE trace_date >= current_date()-INTERVAL 30 DAYS) AS trace_view_cost
""").show(truncate=False)

# COMMAND ----------

# mlflow.trace.cost validation (runs only if the trace_metadata table exists)
META = f"{TRACE_CATALOG}.{TRACE_SCHEMA}.{TABLE_PREFIX}_trace_metadata"
if spark.catalog.tableExists(META):
    spark.sql(f"""
      WITH sample AS (
        SELECT trace_id, mlflow_total_cost FROM {OUT}.v_trace_cost_corrected
        WHERE mlflow_total_cost IS NOT NULL LIMIT 50
      ),
      meta AS (
        SELECT trace_id,
               try_cast(metadata['mlflow.trace.cost'] AS double) AS trace_cost
        FROM {META}
      )
      SELECT count(*) AS n,
             count_if(abs(s.mlflow_total_cost - m.trace_cost) < 1e-6) AS matches,
             round(100*count_if(abs(s.mlflow_total_cost - m.trace_cost) < 1e-6)/count(*),1) AS match_pct
      FROM sample s JOIN meta m USING (trace_id)
    """).show(truncate=False)
else:
    print(f"N/A — {META} does not exist (token-only / AI Gateway tracing has no trace-level cost).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caveat checks
# MAGIC - Implied $/DBU: for spans with mlflow.llm.cost, does cost/token imply $0.07/DBU?
# MAGIC   If not, the flat multiplier is unsafe and the token engine should be used.
# MAGIC - Cache read tokens are inside `input_tokens`; priced on their own rate.
# MAGIC - Spans with tokens but no resolvable model fall to the default rate (flagged).

# COMMAND ----------

# Implied per-token price for mlflow.llm.cost spans (empty on token-only workspaces)
spark.sql(f"""
  SELECT model_key,
         round(sum(mlflow_input_cost_num)/nullif(sum(input_tokens),0)*1e6, 4)  AS usd_per_mtok_in,
         round(sum(mlflow_output_cost_num)/nullif(sum(output_tokens),0)*1e6, 4) AS usd_per_mtok_out
  FROM (
    SELECT model_key, input_tokens, output_tokens,
           attributes:['mlflow.llm.cost']['input_cost']::double  AS mlflow_input_cost_num,
           attributes:['mlflow.llm.cost']['output_cost']::double AS mlflow_output_cost_num
    FROM {OUT}.v_trace_span_cost_corrected v
    JOIN {SPANS} s USING (span_id)
    WHERE v.pricing_method = 'dbu_multiplier'
  ) GROUP BY model_key
""").show(truncate=False)

# COMMAND ----------

# Cache-read share and spans with tokens but no resolvable model
spark.sql(f"""
  SELECT
    round(100*sum(cache_read_tokens)/nullif(sum(input_tokens),0),1) AS pct_input_from_cache,
    count_if(model_key = '(unknown)') AS spans_model_unknown,
    count_if(NOT rate_matched)        AS spans_on_default_rate
  FROM {OUT}.v_trace_span_cost_corrected
  WHERE is_topmost_cost_span AND span_date >= current_date()-INTERVAL 30 DAYS
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Reconciliation vs `system.billing.usage` (FMAPI model-serving SKUs)
# MAGIC Compares corrected daily trace cost against billed DBUs × list price. On a
# MAGIC shared workspace the billed figure spans all serving usage, not just the
# MAGIC traced calls, so expect a large gap on a test bed; on a single-tenant
# MAGIC customer workspace with real rates this shows whether the approach holds.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {OUT}.v_trace_cost_reconciliation AS
WITH corrected AS (
  SELECT trace_date AS usage_date, sum(corrected_cost) AS trace_corrected_cost
  FROM {OUT}.v_trace_cost_corrected GROUP BY 1
),
billed AS (
  SELECT u.usage_date,
    sum(u.usage_quantity * try_cast(lp.pricing.default AS double)) AS billed_list_cost,
    sum(u.usage_quantity) AS billed_dbus
  FROM system.billing.usage u
  JOIN system.billing.list_prices lp
    ON u.sku_name = lp.sku_name
   AND u.usage_end_time >= lp.price_start_time
   AND (lp.price_end_time IS NULL OR u.usage_end_time < lp.price_end_time)
  WHERE u.usage_unit = 'DBU' AND u.sku_name LIKE 'ENTERPRISE_%_MODEL_SERVING'
  GROUP BY 1
)
SELECT coalesce(c.usage_date, b.usage_date) AS usage_date,
  round(c.trace_corrected_cost,2) AS trace_corrected_cost,
  round(b.billed_list_cost,2)     AS billed_list_cost,
  round(b.billed_dbus,1)          AS billed_dbus,
  round(c.trace_corrected_cost - b.billed_list_cost,2) AS gap_usd,
  round(100*(c.trace_corrected_cost - b.billed_list_cost)/nullif(b.billed_list_cost,0),1) AS gap_pct
FROM corrected c FULL OUTER JOIN billed b ON c.usage_date = b.usage_date
""")
spark.sql(f"""
  SELECT * FROM {OUT}.v_trace_cost_reconciliation
  WHERE usage_date >= current_date()-INTERVAL 7 DAYS ORDER BY usage_date
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Teardown (run to remove the overlay when the product fix lands)
# MAGIC Drops only objects in `output_catalog.output_schema`. Never touches the
# MAGIC MLflow-managed trace tables.

# COMMAND ----------

# Uncomment to tear down:
# spark.sql(f"DROP VIEW IF EXISTS {OUT}.v_trace_cost_reconciliation")
# spark.sql(f"DROP VIEW IF EXISTS {OUT}.v_trace_cost_corrected")
# spark.sql(f"DROP VIEW IF EXISTS {OUT}.v_trace_span_cost_corrected")
# spark.sql(f"DROP TABLE IF EXISTS {OUT}.trace_cost_corrected")
# spark.sql(f"DROP TABLE IF EXISTS {OUT}.trace_cost_rate_config")
# spark.sql(f"DROP TABLE IF EXISTS {OUT}.trace_cost_settings")
# spark.sql(f"DROP SCHEMA IF EXISTS {OUT}")
