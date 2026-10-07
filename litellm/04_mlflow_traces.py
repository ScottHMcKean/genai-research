# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · View the ported traces in MLflow
# MAGIC
# MAGIC Points an **MLflow experiment** at the trace rows we already have (litellm/zerobus +
# MAGIC Databricks AI Gateway) so they show up in the **MLflow Traces UI** — one MLflow trace per
# MAGIC request, with the original timestamps, token usage, and messages.
# MAGIC
# MAGIC MLflow traces are OpenTelemetry spans; we bind the experiment to a **Unity Catalog** trace
# MAGIC location (same pattern as `../tracing/01_custom_agent_tracing`), so the ported traces land
# MAGIC in governed Delta tables `{CATALOG}.{SCHEMA}.{MLFLOW_TRACE_PREFIX}_otel_*` too.
# MAGIC
# MAGIC Backfill uses `mlflow.start_span_no_context(start_time_ns=...)` + `span.end(end_time_ns=...)`
# MAGIC — the stable primitive for logging historical spans with their real timestamps.
# MAGIC
# MAGIC > Prereqs: `02_zerobus_logging` (litellm traces) and ideally `03` (gateway sample). Needs
# MAGIC > the **Store OpenTelemetry traces in Unity Catalog** preview enabled (as in tracing/).

# COMMAND ----------

# MAGIC %pip install --quiet -U "mlflow>=3.1"
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
import os, time
from config import (
    CATALOG, SCHEMA, LITELLM_TRACES_TABLE, GATEWAY_INFERENCE_TABLE,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACE_PREFIX, SQL_WAREHOUSE_ID,
)
import trace_adapter as ta

# MLflow needs a SQL warehouse to read/write UC-backed traces (set before binding).
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bind an experiment to a UC trace location

# COMMAND ----------

import mlflow
from mlflow.exceptions import RestException

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

user_email = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_PATH = f"/Users/{user_email}/{MLFLOW_EXPERIMENT_NAME}"

try:
    from mlflow.entities.trace_location import UnityCatalog

    mlflow.set_experiment(
        EXPERIMENT_PATH,
        trace_location=UnityCatalog(
            catalog_name=CATALOG, schema_name=SCHEMA, table_prefix=MLFLOW_TRACE_PREFIX,
        ),
    )
    print(f"Bound {EXPERIMENT_PATH}\n   -> {CATALOG}.{SCHEMA}.{MLFLOW_TRACE_PREFIX}_otel_*")
except RestException as e:
    # A UC trace location can only attach to an experiment with no traces yet.
    if "already" in str(e).lower():
        mlflow.set_experiment(EXPERIMENT_PATH)
        print("Experiment already has a UC trace location — continuing.")
    else:
        raise
except (TypeError, ImportError) as e:
    mlflow.set_experiment(EXPERIMENT_PATH)
    print(f"Could not bind UC location ({type(e).__name__}: {e}). "
          "Traces still log; set the location from the Experiment UI if you want UC storage.")

EXPERIMENT = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
EXPERIMENT_ID = EXPERIMENT.experiment_id
print(f"experiment_id: {EXPERIMENT_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the source rows and normalize
# MAGIC
# MAGIC Reuse the adapter's normalizers so the MLflow trace carries the same unified fields the
# MAGIC SQL view / Python function produce. Reads whichever of the two tables exist.

# COMMAND ----------

unified = []
if spark.catalog.tableExists(LITELLM_TRACES_TABLE):
    rows = [r.asDict() for r in spark.table(LITELLM_TRACES_TABLE).limit(500).collect()]
    unified += [ta.litellm_row_to_unified(r) for r in rows]
    print(f"litellm: {len(rows)} rows")
if spark.catalog.tableExists(GATEWAY_INFERENCE_TABLE):
    rows = [r.asDict() for r in spark.table(GATEWAY_INFERENCE_TABLE).limit(500).collect()]
    unified += [ta.gateway_row_to_unified(r) for r in rows]
    print(f"gateway: {len(rows)} rows")

print(f"total unified traces to port: {len(unified)}")
assert unified, "no source rows found — run 02_zerobus_logging (and 03) first"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Port each row to an MLflow trace (with its real timestamps)

# COMMAND ----------

from datetime import datetime, timezone
from mlflow.entities import SpanType

def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)

ported = 0
for u in unified:
    start = u.start_time or datetime.now(timezone.utc)
    end = u.end_time or (
        datetime.fromtimestamp(start.timestamp() + (u.duration_ms or 0) / 1000.0, tz=timezone.utc)
    )
    span = mlflow.start_span_no_context(
        name=ta.otel_span_name(u),
        span_type=SpanType.LLM,
        inputs={"messages": u.messages},
        attributes=ta.unified_to_otel_attributes(u),
        # user/session metadata so the Traces UI can group + filter (see tracing/01 §c).
        metadata={
            "mlflow.trace.user": u.requester or "unknown",
            "mlflow.trace.session": u.source,          # group by trace source (litellm | gateway)
            "trace.source": u.source,
            "trace.request_id": u.request_id or "",
        },
        experiment_id=EXPERIMENT_ID,
        start_time_ns=_ns(start),
    )
    span.end(
        outputs={"content": u.response_text},
        attributes={
            "gen_ai.usage.total_tokens": u.total_tokens,
            "response.cost": None,  # kept for parity; litellm's response_cost lives on the UC row
        },
        status="OK" if u.status == "success" else "ERROR",
        end_time_ns=_ns(end),
    )
    ported += 1

print(f"ported {ported} traces into experiment {EXPERIMENT_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify they're queryable in MLflow

# COMMAND ----------

def wait_for_traces(experiment_id: str, expected: int, timeout_s: int = 150, interval_s: int = 5) -> int:
    deadline = time.time() + timeout_s
    n = 0
    while time.time() < deadline:
        n = len(mlflow.search_traces(experiment_ids=[experiment_id], return_type="list"))
        if n >= expected:
            break
        time.sleep(interval_s)
    return n

n_ready = wait_for_traces(EXPERIMENT_ID, ported)
print(f"{n_ready}/{ported} traces queryable via mlflow.search_traces\n")

# Per-source breakdown (session == source) and a peek at one trace's spans.
by_source = {}
for t in mlflow.search_traces(experiment_ids=[EXPERIMENT_ID], return_type="list"):
    src = (t.info.trace_metadata or {}).get("mlflow.trace.session", "?")
    by_source[src] = by_source.get(src, 0) + 1
print("traces by source:", by_source)

sample = mlflow.search_traces(experiment_ids=[EXPERIMENT_ID], return_type="list")[0]
spans = mlflow.get_trace(sample.info.trace_id).data.spans
print(f"\nsample trace {sample.info.trace_id}: {len(spans)} span(s)")
for s in spans:
    print(f"  {s.name}  model={s.attributes.get('gen_ai.request.model')}  "
          f"in/out={s.attributes.get('gen_ai.usage.input_tokens')}/{s.attributes.get('gen_ai.usage.output_tokens')}")

# COMMAND ----------

ws = spark.conf.get("spark.databricks.workspaceUrl")
print(f"Open the Traces UI:\n  https://{ws}/ml/experiments/{EXPERIMENT_ID}/traces")
print(f"UC trace tables:    {CATALOG}.{SCHEMA}.{MLFLOW_TRACE_PREFIX}_otel_spans")

try:
    import json
    dbutils.notebook.exit(json.dumps({"experiment_id": EXPERIMENT_ID, "ported": ported, "queryable": n_ready}))
except Exception:
    pass
