# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup — Zerobus prerequisites for the litellm demo
# MAGIC
# MAGIC Prepares everything litellm's **Zerobus logging callback** needs on **shm-skunkworks**:
# MAGIC 1. Creates the `litellm` schema.
# MAGIC 2. Creates the managed Delta **trace table** using litellm's *own* schema generator
# MAGIC    (`litellm.integrations.zerobus.row.create_table_sql`), so the columns always match the
# MAGIC    installed litellm version (40 cols: scalars + VARIANT).
# MAGIC 3. Creates an OAuth **service principal** (`litellm-zerobus-producer`) and grants it
# MAGIC    `USE CATALOG` / `USE SCHEMA` / `MODIFY` + `SELECT` on the trace table. Zerobus
# MAGIC    authenticates as this SP, *not* as the notebook user.
# MAGIC 4. Stashes the SP client id/secret in the `shm` secret scope for 02 to read.
# MAGIC
# MAGIC > Zerobus needs **explicit table-level** `MODIFY`+`SELECT` on the SP (schema-level inherited
# MAGIC > grants are not enough for the OAuth `authorization_details` flow — otherwise you get error 4024).
# MAGIC >
# MAGIC > Idempotent: re-running reuses the existing SP and only mints a new secret if one isn't stored.

# COMMAND ----------

# Serverless base env pins litellm below the Zerobus integration (added in 1.103.0). Clear the
# constraint file so the floor install below actually resolves to a zerobus-capable version.
import os
os.environ["PIP_CONSTRAINT"] = ""

# COMMAND ----------

# MAGIC %pip install -U "litellm==1.105.0rc1" databricks-sdk -q
# MAGIC %restart_python

# COMMAND ----------

import sys; sys.path.append("..") if ".." not in sys.path else None
from config import (
    CATALOG, SCHEMA, LITELLM_TRACES_TABLE, ZEROBUS_SERVER_ENDPOINT, WORKSPACE_URL,
    SP_DISPLAY_NAME, SECRET_SCOPE, SECRET_KEY_CLIENT_ID, SECRET_KEY_CLIENT_SECRET,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Schema + trace table (schema straight from litellm)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Prefer litellm's own DDL so columns always match the installed version. Fall back to the pinned
# 1.103 schema (40 cols) if the integration still isn't importable, so setup never hard-blocks.
try:
    from litellm.integrations.zerobus.row import create_table_sql
    from importlib.metadata import version as _pkgver
    print("using litellm", _pkgver("litellm"), "create_table_sql")
    ddl_raw = create_table_sql(LITELLM_TRACES_TABLE)
except ModuleNotFoundError:
    print("litellm.integrations.zerobus not importable — using fallback DDL (litellm 1.103 schema)")
    _COLS = (
        "id STRING, trace_id STRING, session_id STRING, litellm_call_id STRING, call_type STRING, "
        "status STRING, model STRING, model_group STRING, model_id STRING, custom_llm_provider STRING, "
        "api_base STRING, stream BOOLEAN, cache_hit BOOLEAN, start_time TIMESTAMP, end_time TIMESTAMP, "
        "completion_start_time TIMESTAMP, response_time DOUBLE, prompt_tokens LONG, completion_tokens LONG, "
        "total_tokens LONG, response_cost DOUBLE, saved_cache_cost DOUBLE, api_key_hash STRING, "
        "api_key_alias STRING, team_id STRING, team_alias STRING, user_id STRING, org_id STRING, "
        "end_user STRING, requester_ip_address STRING, user_agent STRING, request_tags VARIANT, "
        "messages VARIANT, response VARIANT, error_str STRING, error_information VARIANT, metadata VARIANT, "
        "model_parameters VARIANT, hidden_params VARIANT, guardrail_information VARIANT, cost_breakdown VARIANT"
    )
    ddl_raw = f"CREATE TABLE {LITELLM_TRACES_TABLE} (\n{_COLS}\n);"

# Make it idempotent for re-runs.
ddl = ddl_raw.replace(
    f"CREATE TABLE {LITELLM_TRACES_TABLE}", f"CREATE TABLE IF NOT EXISTS {LITELLM_TRACES_TABLE}"
)
print(ddl)
spark.sql(ddl)
print("\ntrace table ready:", LITELLM_TRACES_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Service principal + OAuth secret
# MAGIC
# MAGIC Reuses the SP if it already exists. Mints a secret only if one isn't already stored in the scope.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Find-or-create the service principal by display name.
sp = next((s for s in w.service_principals.list() if s.display_name == SP_DISPLAY_NAME), None)
if sp is None:
    sp = w.service_principals.create(display_name=SP_DISPLAY_NAME)
    print("created service principal:", SP_DISPLAY_NAME)
else:
    print("reusing service principal:", SP_DISPLAY_NAME)

client_id = sp.application_id   # OAuth client id (the UC principal identifier for SPs)
print("application_id (client_id):", client_id, "| sp id:", sp.id)

# COMMAND ----------

# Ensure the secret scope exists, then store creds. We can't re-read an old OAuth secret, so if a
# client_secret isn't already stored, mint a fresh one and store both id + secret.
scopes = {s.name for s in w.secrets.list_scopes()}
if SECRET_SCOPE not in scopes:
    w.secrets.create_scope(scope=SECRET_SCOPE)
    print("created secret scope:", SECRET_SCOPE)

existing = {s.key for s in w.secrets.list_secrets(scope=SECRET_SCOPE)}
if SECRET_KEY_CLIENT_SECRET not in existing:
    # Workspace-level SP OAuth secret -> service_principal_secrets_proxy (plain
    # `service_principal_secrets` is account-scoped and absent on WorkspaceClient).
    secret = w.service_principal_secrets_proxy.create(service_principal_id=str(sp.id))
    w.secrets.put_secret(scope=SECRET_SCOPE, key=SECRET_KEY_CLIENT_ID, string_value=client_id)
    w.secrets.put_secret(scope=SECRET_SCOPE, key=SECRET_KEY_CLIENT_SECRET, string_value=secret.secret)
    print(f"minted + stored OAuth secret -> {SECRET_SCOPE}/{SECRET_KEY_CLIENT_SECRET}")
else:
    # Keep client_id in sync in case the SP was recreated.
    w.secrets.put_secret(scope=SECRET_SCOPE, key=SECRET_KEY_CLIENT_ID, string_value=client_id)
    print(f"OAuth secret already stored at {SECRET_SCOPE}/{SECRET_KEY_CLIENT_SECRET} (left as-is)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Grant the SP write access to the trace table
# MAGIC
# MAGIC Explicit table-level `MODIFY` + `SELECT` (required by the Zerobus OAuth flow), plus the
# MAGIC `USE` grants up the hierarchy.

# COMMAND ----------

for stmt in (
    f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{client_id}`",
    f"GRANT USE SCHEMA ON SCHEMA {CATALOG}.{SCHEMA} TO `{client_id}`",
    f"GRANT MODIFY, SELECT ON TABLE {LITELLM_TRACES_TABLE} TO `{client_id}`",
):
    spark.sql(stmt)
    print("ok:", stmt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Summary — the five values litellm's zerobus callback needs

# COMMAND ----------

print("ZEROBUS_WORKSPACE_URL   :", WORKSPACE_URL)
print("ZEROBUS_SERVER_ENDPOINT :", ZEROBUS_SERVER_ENDPOINT)
print("ZEROBUS_TABLE_NAME      :", LITELLM_TRACES_TABLE)
print("ZEROBUS_CLIENT_ID       :", client_id)
print(f"ZEROBUS_CLIENT_SECRET   : (stored at {SECRET_SCOPE}/{SECRET_KEY_CLIENT_SECRET})")
print("\nSetup complete. Run 01_litellm_sdk_test next, then 02_zerobus_logging.")
