# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Custom RAG agent — log, register, deploy
# MAGIC
# MAGIC Wraps the Vector Search retriever + a foundation model as an MLflow
# MAGIC **ResponsesAgent** (`agent.py`), logs it with the right resources for auth
# MAGIC passthrough, registers it to Unity Catalog, and deploys a **Model Serving**
# MAGIC endpoint with a review app.
# MAGIC
# MAGIC Showcases: MLflow 3 agent authoring, UC model registry, one-line agent
# MAGIC deployment (`agents.deploy`), automatic auth to VS + FMAPI.

# COMMAND ----------

# MAGIC %pip install --quiet -U mlflow databricks-langchain databricks-vectorsearch databricks-agents
# MAGIC %restart_python

# COMMAND ----------

from config import RAG_UC_MODEL, RAG_ENDPOINT, VS_INDEX
from agent import AGENT, LLM_ENDPOINT, EMBED_ENDPOINT
from mlflow.types.responses import ResponsesAgentRequest

# quick local smoke test before logging
req = ResponsesAgentRequest(input=[{"role": "user",
    "content": "When can a glass-only auto claim be settled without an inspection?"}])
print(AGENT.predict(req).output[0].content[:600])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log + register to Unity Catalog

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

mlflow.set_registry_uri("databricks-uc")

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=EMBED_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=VS_INDEX),
]

with mlflow.start_run(run_name="claims_rag_agent"):
    info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        code_paths=["config.py"],   # agent.py imports config at serve time
        resources=resources,
        pip_requirements=[
            "mlflow",
            "databricks-langchain",
            "databricks-vectorsearch",
        ],
        input_example={"input": [{"role": "user", "content": "What is covered under sewer backup?"}]},
        registered_model_name=RAG_UC_MODEL,
    )
print("Logged + registered:", info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Model Serving (~15 min to READY)

# COMMAND ----------

from databricks import agents

# agents.deploy signature is deploy(model_name, model_version, ...) -- pass version positionally.
deployment = agents.deploy(
    RAG_UC_MODEL,
    info.registered_model_version,
    endpoint_name=RAG_ENDPOINT,
    tags={"demo": "claims", "type": "custom_rag"},
)
print("Deploying endpoint:", RAG_ENDPOINT)
print("Review app / query URL will appear in the endpoint page once READY.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the deployed agent (run after the endpoint is READY)

# COMMAND ----------

# from mlflow.deployments import get_deploy_client
# client = get_deploy_client("databricks")
# r = client.predict(endpoint=RAG_ENDPOINT, inputs={"input": [
#     {"role": "user", "content": "A visitor slipped on icy steps and went to hospital. How should this be triaged?"}]})
# print(r["output"][0]["content"][0]["text"])
