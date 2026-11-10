# Databricks notebook source
# MAGIC %md
# MAGIC # Materialize corrected trace cost (incremental, daily)
# MAGIC
# MAGIC Incrementally writes `v_trace_cost_corrected` into the Delta table
# MAGIC `output.trace_cost_corrected`, keyed on `trace_id`. Uses a watermark on
# MAGIC `trace_start_time` with a lookback window so late-arriving spans within a
# MAGIC trace are picked up. **MERGE targets our table only** — the MLflow-managed
# MAGIC trace tables are read-only upstream of the view.

# COMMAND ----------

dbutils.widgets.text("output_catalog", "uaigw_fe", "Output catalog")
dbutils.widgets.text("output_schema", "trace_cost_overlay", "Output schema")
dbutils.widgets.text("lookback_hours", "48", "Watermark lookback (hours)")

OUT = f'{dbutils.widgets.get("output_catalog")}.{dbutils.widgets.get("output_schema")}'
LOOKBACK = int(dbutils.widgets.get("lookback_hours"))
TARGET = f"{OUT}.trace_cost_corrected"

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {TARGET} (
    trace_id STRING,
    trace_date DATE,
    trace_start_time TIMESTAMP,
    trace_end_time TIMESTAMP,
    llm_span_count BIGINT,
    input_tokens BIGINT,
    cache_read_tokens BIGINT,
    billable_input_tokens BIGINT,
    output_tokens BIGINT,
    mlflow_total_cost DOUBLE,
    corrected_cost DOUBLE,
    pricing_mode STRING,
    all_rates_matched BOOLEAN,
    unmatched_span_count BIGINT,
    models STRING,
    pricing_methods STRING,
    _materialized_at TIMESTAMP
  )
  USING DELTA
  PARTITIONED BY (trace_date)
  COMMENT 'Materialized corrected trace cost. Keyed on trace_id. Rebuilds safely; safe to drop.'
""")

# COMMAND ----------

# Watermark = latest trace_start_time already materialized, minus lookback.
wm = spark.sql(f"SELECT max(trace_start_time) FROM {TARGET}").collect()[0][0]
if wm is None:
    where = "trace_start_time IS NOT NULL"
    print("Cold start: full backfill.")
else:
    where = f"trace_start_time >= (TIMESTAMP'{wm}' - INTERVAL {LOOKBACK} HOURS)"
    print(f"Incremental from watermark {wm} minus {LOOKBACK}h.")

# COMMAND ----------

spark.sql(f"""
  MERGE INTO {TARGET} t
  USING (
    SELECT *, current_timestamp() AS _materialized_at
    FROM {OUT}.v_trace_cost_corrected
    WHERE {where}
  ) s
  ON t.trace_id = s.trace_id
  WHEN MATCHED THEN UPDATE SET *
  WHEN NOT MATCHED THEN INSERT *
""")

n = spark.sql(f"SELECT count(*) FROM {TARGET}").collect()[0][0]
print(f"{TARGET} now has {n} rows.")
dbutils.notebook.exit(f"ok:{n}")
