# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · MCP on Databricks — govern the tools your agents use
# MAGIC
# MAGIC MCP (Model Context Protocol) is how agents call tools. On Databricks, Unity Catalog
# MAGIC exposes **managed MCP servers** so your governed assets *are* the tools — no glue code,
# MAGIC and every call runs with the **caller's own permissions** (on-behalf-of-user).
# MAGIC
# MAGIC | Managed MCP server | URL | Governs |
# MAGIC |---|---|---|
# MAGIC | UC Functions | `/api/2.0/mcp/functions/{catalog}/{schema}` | SQL/Python functions as tools |
# MAGIC | Vector Search | `/api/2.0/mcp/vector-search/{catalog}/{schema}` | Indexes as retrieval tools |
# MAGIC | Genie | `/api/2.0/mcp/genie/{space_id}` | A Genie space as a tool |
# MAGIC
# MAGIC We register two claims **UC functions**, expose them as MCP tools, and show how an
# MAGIC **external** MCP server is governed through a UC connection.

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-sdk mcp databricks-mcp
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".")
from config import CATALOG, SCHEMA, CLAIMS_TABLE, LOOKUP_FUNCTION, MASK_FUNCTION

host = spark.conf.get("spark.databricks.workspaceUrl")
print("Workspace:", host)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Register claims tools as UC functions
# MAGIC Governed like any table: `GRANT EXECUTE` decides who (and which agent) can call them.

# COMMAND ----------

# Tool 1: look up a claim by id (returns a compact summary; no PII columns exposed)
spark.sql(f"""
CREATE OR REPLACE FUNCTION {LOOKUP_FUNCTION}(p_claim_id STRING)
RETURNS TABLE(claim_id STRING, claim_type STRING, peril STRING,
              claim_status STRING, severity STRING, claim_amount DOUBLE, region STRING)
COMMENT 'Look up a claim summary by claim_id. Exposed as an MCP tool for claims agents.'
RETURN SELECT claim_id, claim_type, peril, claim_status, severity, claim_amount, region
       FROM {CLAIMS_TABLE} WHERE claim_id = p_claim_id
""")

# Tool 2: deterministic SSN mask (governance helper an agent can call)
spark.sql(f"""
CREATE OR REPLACE FUNCTION {MASK_FUNCTION}(p_ssn STRING)
RETURNS STRING
COMMENT 'Mask an SSN-shaped string, keeping only the last 4 digits. MCP-exposed governance tool.'
RETURN CASE WHEN p_ssn IS NULL THEN NULL
            ELSE concat('XXX-XX-', substring(regexp_replace(p_ssn,'[^0-9]',''), -4, 4)) END
""")
print("Registered UC functions (now MCP tools):")
print(" -", LOOKUP_FUNCTION)
print(" -", MASK_FUNCTION)

# COMMAND ----------

# quick functional test
display(spark.sql(f"SELECT {MASK_FUNCTION}('123-45-6789') AS masked"))
display(spark.sql(f"SELECT * FROM {LOOKUP_FUNCTION}((SELECT claim_id FROM {CLAIMS_TABLE} LIMIT 1))"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · These functions are now a managed MCP server
# MAGIC No server to run — Unity Catalog hosts it. Point any MCP client at the URL below and
# MAGIC the two functions appear as tools, callable **with the user's permissions**.

# COMMAND ----------

functions_mcp_url = f"https://{host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"
vs_mcp_url        = f"https://{host}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"
print("UC Functions MCP server :", functions_mcp_url)
print("Vector Search MCP server:", vs_mcp_url)
print("Genie MCP server        :", f"https://{host}/api/2.0/mcp/genie/<genie_space_id>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Connect to the managed MCP server (on-behalf-of-user)
# MAGIC The Databricks MCP client authenticates as the caller — tool calls inherit UC RBAC.

# COMMAND ----------

try:
    from databricks.sdk import WorkspaceClient
    from databricks_mcp import DatabricksMCPClient   # ships in `databricks-mcp`

    w = WorkspaceClient()
    mcp = DatabricksMCPClient(server_url=functions_mcp_url, workspace_client=w)
    tools = mcp.list_tools()
    print("Tools exposed by the managed MCP server:")
    for t in tools:
        print(" -", t.name)
except Exception as e:
    print("MCP client listing skipped:", type(e).__name__, str(e)[:200])
    print("Fallback: the URL above works from any MCP client (Claude, custom agent, "
          "or Agent Bricks 'External MCP' connection). See docs: /generative-ai/mcp.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Govern an *external* MCP server via a UC connection
# MAGIC Third-party MCP servers (e.g. a policy-admin system) are registered as a Unity Catalog
# MAGIC **connection** with `is_mcp_connection='true'`. Credentials live in UC (not in agent
# MAGIC code), and `USE CONNECTION` grants control who can reach it.

# COMMAND ----------

# Conceptual DDL -- create only if you have a real external MCP endpoint + token.
external_mcp_ddl = f"""
CREATE CONNECTION IF NOT EXISTS claims_external_mcp
  TYPE HTTP
  OPTIONS (
    host 'https://your-external-mcp.example.com',
    port '443',
    base_path '/mcp',
    bearer_token secret('claims_demo','external_mcp_token'),
    is_mcp_connection 'true'
  )
"""
print(external_mcp_ddl)
print("Then: GRANT USE CONNECTION ON CONNECTION claims_external_mcp TO `claims-agents`;")
# Not executed here (no real external endpoint) -- shown so the audience sees the governance pattern.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Talk track
# MAGIC - **Your governed assets are the tools.** A UC function, a Vector Search index, a Genie
# MAGIC   space — all become MCP tools with zero glue and full lineage.
# MAGIC - **On-behalf-of-user by default.** An agent can never see data the *caller* can't.
# MAGIC - **External tools, same governance.** Third-party MCP servers are gated by UC
# MAGIC   connections and `USE CONNECTION` grants — credentials never touch agent code.
