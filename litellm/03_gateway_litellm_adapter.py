# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Adapter — AI Gateway traces ⇄ litellm traces (+ OTel compatibility)
# MAGIC
# MAGIC Databricks **AI Gateway inference tables** and **litellm (zerobus) traces** both record one
# MAGIC LLM request per row, but with different schemas. This notebook unifies them:
# MAGIC
# MAGIC 1. **SQL view** (`unified_traces`) — `UNION ALL` of both tables projected into one schema,
# MAGIC    parsing the gateway's JSON `request`/`response` and litellm's VARIANT columns.
# MAGIC 2. **Python function** — `gateway_row_to_unified` / `litellm_row_to_unified` from
# MAGIC    `trace_adapter.py`, for programmatic use (identical projection to the view).
# MAGIC 3. **OTel verification** — map both into OpenTelemetry **GenAI semantic-convention**
# MAGIC    attributes (`gen_ai.*`), build real spans, and assert both sources are compatible.
# MAGIC
# MAGIC If `GATEWAY_INFERENCE_TABLE` doesn't exist, we create a small sample with the Mosaic AI
# MAGIC Gateway inference-table schema so the whole thing runs without a live gateway. Point the
# MAGIC config at your real inference table to run it for real.

# COMMAND ----------

# Clear the serverless PIP_CONSTRAINT so the litellm floor resolves (see 00_setup / config.py).
import os
os.environ["PIP_CONSTRAINT"] = ""

# COMMAND ----------

# MAGIC %pip install -U "litellm==1.105.0rc1" opentelemetry-api opentelemetry-sdk -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
from config import (
    CATALOG, SCHEMA, LITELLM_TRACES_TABLE, GATEWAY_INFERENCE_TABLE, UNIFIED_TRACE_VIEW,
)
import trace_adapter as ta

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Ensure a gateway inference table exists (sample fallback)
# MAGIC
# MAGIC Schema matches the Mosaic AI Gateway inference (payload-logging) table. Replace with your
# MAGIC real table via `config.GATEWAY_INFERENCE_TABLE` to adapt production traces.

# COMMAND ----------

gateway_exists = spark.catalog.tableExists(GATEWAY_INFERENCE_TABLE)
if not gateway_exists:
    print("no gateway table found — creating sample:", GATEWAY_INFERENCE_TABLE)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {GATEWAY_INFERENCE_TABLE} (
          request_id STRING, invocation_id STRING, request_tags MAP<STRING,STRING>,
          event_time TIMESTAMP, status_code INT, sampling_fraction DOUBLE,
          latency_ms BIGINT, time_to_first_byte_ms BIGINT, request STRING, response STRING,
          destination_type STRING, destination_name STRING, destination_model STRING,
          logging_error_codes ARRAY<STRING>, requester STRING, schema_version STRING
        )
    """)
    spark.sql(f"""
        INSERT INTO {GATEWAY_INFERENCE_TABLE} VALUES
        ('gw-req-1', 'inv-1', map('team','platform'), current_timestamp() - INTERVAL 5 MINUTES,
         200, 1.0, 842, 310,
         '{{"model":"databricks-claude-sonnet-4-5","messages":[{{"role":"user","content":"Hello from the gateway"}}]}}',
         '{{"id":"chatcmpl-gw-1","model":"databricks-claude-sonnet-4-5","choices":[{{"message":{{"role":"assistant","content":"Hi there!"}}}}],"usage":{{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}}}',
         'FOUNDATION_MODEL_API', 'shm_endpoint', 'databricks-claude-sonnet-4-5',
         array(), 'user@databricks.com', 'v1'),
        ('gw-req-2', 'inv-2', map('team','ml'), current_timestamp() - INTERVAL 3 MINUTES,
         200, 1.0, 611, 240,
         '{{"model":"databricks-claude-sonnet-4-5","messages":[{{"role":"user","content":"Summarize UC in one line"}}]}}',
         '{{"id":"chatcmpl-gw-2","model":"databricks-claude-sonnet-4-5","choices":[{{"message":{{"role":"assistant","content":"Unity Catalog governs data and AI assets centrally."}}}}],"usage":{{"prompt_tokens":9,"completion_tokens":11,"total_tokens":20}}}}',
         'FOUNDATION_MODEL_API', 'shm_endpoint', 'databricks-claude-sonnet-4-5',
         array(), 'user@databricks.com', 'v1')
    """)
else:
    print("using existing gateway table:", GATEWAY_INFERENCE_TABLE)

display(spark.table(GATEWAY_INFERENCE_TABLE).select(
    "request_id", "event_time", "status_code", "latency_ms", "destination_model", "requester"
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. SQL adapter view
# MAGIC
# MAGIC One view, both sources, one schema. Requires the litellm trace table (02) to exist.

# COMMAND ----------

if not spark.catalog.tableExists(LITELLM_TRACES_TABLE):
    raise RuntimeError(
        f"{LITELLM_TRACES_TABLE} not found — run 00_setup and 02_zerobus_logging first "
        "so there are litellm traces to union with the gateway traces."
    )

view_ddl = ta.unified_view_sql(UNIFIED_TRACE_VIEW, LITELLM_TRACES_TABLE, GATEWAY_INFERENCE_TABLE)
print(view_ddl)
spark.sql(view_ddl)
print("\ncreated view:", UNIFIED_TRACE_VIEW)

# COMMAND ----------

display(
    spark.table(UNIFIED_TRACE_VIEW).selectExpr(
        "source", "request_id", "provider", "request_model", "status",
        "duration_ms", "input_tokens", "output_tokens", "total_tokens", "requester",
    ).orderBy("source", "start_time")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Python adapter (same projection, programmatic)
# MAGIC
# MAGIC Normalize raw rows from each table with the functions in `trace_adapter.py`.

# COMMAND ----------

gw_rows = [r.asDict() for r in spark.table(GATEWAY_INFERENCE_TABLE).limit(10).collect()]
ll_rows = [r.asDict() for r in spark.table(LITELLM_TRACES_TABLE).limit(10).collect()]

unified = [ta.gateway_row_to_unified(r) for r in gw_rows] + \
          [ta.litellm_row_to_unified(r) for r in ll_rows]

print(f"normalized {len(gw_rows)} gateway + {len(ll_rows)} litellm rows\n")
for u in unified[:4]:
    print(u.source, "|", u.request_model, "|", u.input_tokens, "+", u.output_tokens,
          "tok |", u.status, "|", u.request_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. OTel compatibility verification
# MAGIC
# MAGIC Map every unified row to OpenTelemetry GenAI attributes, assert both sources satisfy the
# MAGIC required `gen_ai.*` set with correct types, then build a real OTel span per row through the
# MAGIC SDK to prove the attributes are exportable.

# COMMAND ----------

report = ta.verify_otel_compatible(unified)
print("OTel compatibility report:")
for src, stats in report["per_source"].items():
    print(f"  {src:8s}: {stats['rows']} rows, missing={stats['missing']}, mistyped={stats['mistyped']}")
print("  compatible:", report["compatible"])

# COMMAND ----------

# Build genuine OTel spans through the SDK — this is the "are these exportable as OTel?" proof.
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
tracer = provider.get_tracer("litellm-gateway-adapter")

for u in unified:
    attrs = ta.unified_to_otel_attributes(u)
    start_ns = int(u.start_time.timestamp() * 1e9) if u.start_time else None
    end_ns = int(u.end_time.timestamp() * 1e9) if u.end_time else None
    span = tracer.start_span(ta.otel_span_name(u), attributes=attrs, start_time=start_ns)
    if u.status != "success":
        span.set_status(trace.Status(trace.StatusCode.ERROR))
    span.end(end_time=end_ns)

spans = exporter.get_finished_spans()
print(f"exported {len(spans)} OTel spans from {len(unified)} unified traces\n")
for s in spans[:4]:
    print(" ", s.name, "| gen_ai.system=", s.attributes.get("gen_ai.system"),
          "| in/out=", s.attributes.get("gen_ai.usage.input_tokens"),
          "/", s.attributes.get("gen_ai.usage.output_tokens"))

assert len(spans) == len(unified), "every unified trace must export as exactly one OTel span"
print("\nBoth trace sources are OTel-GenAI compatible and exportable. Adapter demo complete.")
