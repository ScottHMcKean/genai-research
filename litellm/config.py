# Config for the LiteLLM + Databricks Zerobus demo on shm-skunkworks.
# Kept local so the folder is self-contained when synced as a bundle (mirrors pii_serving/config.py).
#
# What this demo exercises:
#   1. The litellm Python package against Databricks Foundation Model APIs  (01_litellm_sdk_test)
#   2. litellm's Zerobus logging callback -> a UC-managed Delta trace table  (02_zerobus_logging)
#   3. An adapter between Databricks AI Gateway inference tables and the      (03_gateway_litellm_adapter)
#      litellm trace table, plus an OpenTelemetry GenAI compatibility check.

# --- Workspace (shm-skunkworks / AWS us-west-2) ---
WORKSPACE_URL = "https://fevm-shm-skunkworks.cloud.databricks.com"
WORKSPACE_ID = "7474644262257186"
REGION = "us-west-2"

# Zerobus REST ingest endpoint. Format (AWS): https://<workspace-id>.zerobus.<region>.cloud.databricks.com
# litellm POSTs batches to {server_endpoint}/zerobus/v1/tables/<table>/insert
ZEROBUS_SERVER_ENDPOINT = f"https://{WORKSPACE_ID}.zerobus.{REGION}.cloud.databricks.com"

# --- Unity Catalog ---
CATALOG = "shm_skunkworks_catalog"   # workspace-local catalog (same one the other demos use)
SCHEMA = "litellm"

# Destination for litellm's zerobus callback. Managed Delta table; schema generated from
# litellm.integrations.zerobus.row (40 columns, scalar + VARIANT). Created by 00_setup.
LITELLM_TRACES_TABLE = f"{CATALOG}.{SCHEMA}.litellm_traces"

# AI Gateway inference (payload-logging) table to adapt against. Point this at your real
# Mosaic AI Gateway inference table; if it doesn't exist, 03 creates a small sample with
# this name/schema so the adapter + OTel check run end-to-end without a live gateway.
GATEWAY_INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.gateway_inference_sample"

# Unified view produced by the adapter (UNION of both trace sources in one schema).
UNIFIED_TRACE_VIEW = f"{CATALOG}.{SCHEMA}.unified_traces"

# --- MLflow experiment for viewing the ported traces (04_mlflow_traces) ---
# 04 backfills the litellm + gateway rows as MLflow traces so they show in the Traces UI,
# stored in UC like the tracing/ demo ({CATALOG}.{SCHEMA}.{MLFLOW_TRACE_PREFIX}_otel_*).
MLFLOW_EXPERIMENT_NAME = "litellm_ported_traces"
MLFLOW_TRACE_PREFIX = "litellm_mlflow"

# --- Models (Databricks FMAPI served-endpoint names; litellm uses the 'databricks/' provider) ---
CHAT_MODEL = "databricks-claude-sonnet-4-5"
EMBED_MODEL = "databricks-gte-large-en"   # optional embeddings smoke test (skipped if absent)

# --- Service principal for Zerobus OAuth (M2M) ---
# Zerobus authenticates as an OAuth service principal, NOT the notebook user. 00_setup creates
# the SP, grants it MODIFY+SELECT on the trace table, and stashes the client id/secret here.
SP_DISPLAY_NAME = "litellm-zerobus-producer"
SECRET_SCOPE = "shm"
SECRET_KEY_CLIENT_ID = "litellm_zb_client_id"
SECRET_KEY_CLIENT_SECRET = "litellm_zb_secret"

# --- SQL warehouse (Serverless Starter) ---
SQL_WAREHOUSE_ID = "505ec857e6b4ea23"

# --- Pins ---
# The zerobus integration (litellm.integrations.zerobus) merged 2026-09-25 but is NOT in any stable
# release yet -- stable 1.104.0 predates the merge. It first ships in v1.105.0-rc.1, so we pin that
# pre-release explicitly. (Also: serverless base envs ship a PIP_CONSTRAINT file that caps litellm;
# the notebooks clear PIP_CONSTRAINT before installing so this pin actually takes effect.)
# Bump to the 1.105.0 stable once it's published.
LITELLM_VERSION = "1.105.0rc1"
LITELLM_PIP = f"litellm=={LITELLM_VERSION}"
OTEL_PIP = "opentelemetry-api>=1.27.0 opentelemetry-sdk>=1.27.0"
