# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Custom RAG agent — Vector Search via MCP, logged & deployed
# MAGIC
# MAGIC The agent (`agent.py`) is an MLflow **ResponsesAgent** that retrieves from the claims
# MAGIC knowledge base by calling the **managed Vector Search MCP server**
# MAGIC (`/api/2.0/mcp/vector-search/{catalog}/{schema}`) as a tool. The LLM decides when to
# MAGIC retrieve; the agent returns the intermediate tool-call + tool-output steps alongside
# MAGIC the final answer (so a chat UI can show its "thinking").
# MAGIC
# MAGIC This notebook: smoke-test → log + register (with MCP-derived resources) → deploy →
# MAGIC **populate 10 traces** → query. Showcases MLflow 3 agent authoring, MCP tool calling,
# MAGIC UC model registry, one-line `agents.deploy`, and auto tracing.

# COMMAND ----------

# MAGIC %pip install --quiet -U mlflow databricks-agents databricks-mcp "databricks-sdk[openai]" databricks-openai mcp
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Smoke test — shows the MCP tool calls + the agent's answer
import mlflow
from agent import AGENT, LLM_ENDPOINT, VS_MCP_PATH
from config import RAG_UC_MODEL, RAG_ENDPOINT
from mlflow.types.responses import ResponsesAgentRequest

resp = AGENT.predict(ResponsesAgentRequest(input=[{"role": "user",
    "content": "What is the procedure for a basement water damage claim, and is sewer backup covered?"}]))

for it in resp.model_dump(exclude_none=True)["output"]:
    t = it.get("type")
    if t == "function_call":
        print(f"  → tool call: {it['name']}({it['arguments']})")
    elif t == "function_call_output":
        print(f"  ← tool result: {it['output'][:120].strip()}...")
    elif t == "message":
        c = it["content"]
        print("\nAnswer:\n" + (c[0]["text"] if isinstance(c, list) else c))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log + register to Unity Catalog
# MAGIC
# MAGIC Resources are derived from the MCP server itself via
# MAGIC `DatabricksMCPClient.get_databricks_resources()` (returns the underlying Vector Search
# MAGIC index) plus the chat endpoint — so the deployed agent gets automatic auth to both.

# COMMAND ----------

import concurrent.futures
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from mlflow.models.resources import DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")
ws = WorkspaceClient()
vs_mcp_url = ws.config.host.rstrip("/") + VS_MCP_PATH

# get_databricks_resources uses asyncio internally -> run in a worker thread in notebooks.
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
    mcp_resources = _ex.submit(
        DatabricksMCPClient(server_url=vs_mcp_url, workspace_client=ws).get_databricks_resources
    ).result()

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT), *mcp_resources]
print("Resources:", resources)

with mlflow.start_run(run_name="claims_rag_agent"):
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        code_paths=["config.py"],
        resources=resources,
        pip_requirements=[
            "mlflow", "databricks-agents", "databricks-mcp",
            "databricks-sdk[openai]", "databricks-openai", "mcp",
        ],
        input_example={"input": [{"role": "user", "content": "Is sewer backup covered?"}]},
        registered_model_name=RAG_UC_MODEL,
    )
print("Logged + registered:", info.model_uri, "v" + str(info.registered_model_version))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Model Serving (~15 min to READY)

# COMMAND ----------

from databricks import agents

# agents.deploy(model_name, model_version, ...) -- version is positional.
agents.deploy(
    RAG_UC_MODEL,
    info.registered_model_version,
    endpoint_name=RAG_ENDPOINT,
    tags={"demo": "claims", "type": "custom_rag_mcp"},
)
print("Deploying endpoint:", RAG_ENDPOINT, "(READY in ~15 min)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Populate traces — run 10 questions through the agent
# MAGIC
# MAGIC Fires 10 realistic claims questions so the MLflow **Traces** UI has content to
# MAGIC explore (each trace shows the MCP Vector Search tool call + the LLM answer).

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/claims_rag_agent")

QUESTIONS = [
    "What is the procedure for a basement water damage claim?",
    "Is sewer backup covered under the homeowners policy?",
    "When can a glass-only auto claim be settled without an inspection?",
    "What fraud indicators require a referral to the Special Investigations Unit?",
    "What triggers escalation to a senior adjuster?",
    "How are Additional Living Expenses handled after a house fire?",
    "What is the insured's duty to mitigate after water damage?",
    "When should an auto claim be declared a total loss?",
    "What is the catastrophe (CAT) response procedure for claims?",
    "How is a stolen financed vehicle settled?",
]

for i, q in enumerate(QUESTIONS, 1):
    with mlflow.start_span(name=f"claims_q_{i:02d}") as span:
        span.set_inputs({"question": q})
        r = AGENT.predict(ResponsesAgentRequest(input=[{"role": "user", "content": q}]))
        msg = [o for o in r.model_dump(exclude_none=True)["output"] if o.get("type") == "message"]
        ans = (msg[-1]["content"][0]["text"] if msg and isinstance(msg[-1]["content"], list)
               else (msg[-1]["content"] if msg else ""))
        span.set_outputs({"answer": ans})
    print(f"[{i:02d}] {q}")
print(f"\n{len(QUESTIONS)} traces written to /Users/{user}/claims_rag_agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the deployed agent (run after the endpoint is READY)

# COMMAND ----------

# from mlflow.deployments import get_deploy_client
# client = get_deploy_client("databricks")
# r = client.predict(endpoint=RAG_ENDPOINT, inputs={"input": [
#     {"role": "user", "content": "A visitor slipped on icy steps and went to hospital. How should this be triaged?"}]})
# print(r["output"][-1]["content"][0]["text"])
