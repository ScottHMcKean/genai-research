# Databricks notebook source
# MAGIC %md
# MAGIC # Materialize corrected trace cost (incremental, daily)
# MAGIC
# MAGIC Incrementally writes `v_trace_cost_corrected` into the Delta table
# MAGIC `output.trace_cost_corrected`, keyed on `trace_id`, with a watermark on
# MAGIC `trace_date` and a lookback. MERGE targets our table only; the system cost
# MAGIC tables and MLflow trace tables are read-only upstream of the view.
# MAGIC
# MAGIC The target schema is taken from the view (`CREATE TABLE AS ... WHERE false`),
# MAGIC so this job works for both `gateway` and `mlflow_span` trace modes.

# COMMAND ----------

dbutils.widgets.text("output_catalog", "uaigw_fe", "Output catalog")
dbutils.widgets.text("output_schema", "trace_cost_overlay", "Output schema")
dbutils.widgets.text("lookback_days", "3", "Watermark lookback (days)")

OUT = f'{dbutils.widgets.get("output_catalog")}.{dbutils.widgets.get("output_schema")}'
LOOKBACK = int(dbutils.widgets.get("lookback_days"))
TARGET = f"{OUT}.trace_cost_corrected"
SOURCE = f"{OUT}.v_trace_cost_corrected"

# COMMAND ----------

# Create the target from the view's schema on first run (plus a load timestamp).
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {TARGET}
  USING DELTA
  PARTITIONED BY (trace_date)
  AS SELECT *, current_timestamp() AS _materialized_at FROM {SOURCE} WHERE 1=0
""")

# COMMAND ----------

wm = spark.sql(f"SELECT max(trace_date) FROM {TARGET}").collect()[0][0]
if wm is None:
    where = "trace_date IS NOT NULL"
    print("Cold start: full backfill.")
else:
    where = f"trace_date >= (DATE'{wm}' - INTERVAL {LOOKBACK} DAYS)"
    print(f"Incremental from watermark {wm} minus {LOOKBACK}d.")

# COMMAND ----------

spark.sql(f"""
  MERGE INTO {TARGET} t
  USING (SELECT *, current_timestamp() AS _materialized_at FROM {SOURCE} WHERE {where}) s
  ON t.trace_id = s.trace_id
  WHEN MATCHED THEN UPDATE SET *
  WHEN NOT MATCHED THEN INSERT *
""")
n = spark.sql(f"SELECT count(*) FROM {TARGET}").collect()[0][0]
print(f"{TARGET} now has {n} rows.")
dbutils.notebook.exit(f"ok:{n}")
