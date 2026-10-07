# Config for the custom-agent tracing demo (Monte Carlo forecasting chat).
# Kept local so the folder is self-contained when synced as a bundle.

CATALOG = "shm_skunkworks_catalog"
SCHEMA = "forecast_demo"

CHAT_MODEL = "databricks-claude-sonnet-4-5"

# MLflow experiment the traces are written to (one per user; set at runtime).
# Keep this distinct from any workspace folder name at the same path.
EXPERIMENT_NAME = "forecast_tracing_exp"

# UC trace location. OTel spans land in Delta tables named
#   {CATALOG}.{SCHEMA}.{TRACE_TABLE_PREFIX}_otel_spans  (+ _otel_logs, ...)
# governed by Unity Catalog like any other table.
TRACE_TABLE_PREFIX = "forecast_traces"
TRACE_SPANS_TABLE = f"{CATALOG}.{SCHEMA}.{TRACE_TABLE_PREFIX}_otel_spans"

# SQL warehouse used by MLflow to write/read UC-backed traces. Required to bind an
# experiment to a UC trace location (set as MLFLOW_TRACING_SQL_WAREHOUSE_ID).
SQL_WAREHOUSE_ID = "505ec857e6b4ea23"

# Group that always sees every trace (secure-view bypass for admins/auditors).
TRACE_ADMIN_GROUP = "trace-admins"

# Registered UC model + serving endpoint if you deploy the agent.
UC_MODEL = f"{CATALOG}.{SCHEMA}.forecast_chat_agent"
ENDPOINT = "forecast-chat-agent"
