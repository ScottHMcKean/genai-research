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

# DBTITLE 1,Local smoke test (shows VS call + agent response)
import mlflow, textwrap
from config import RAG_UC_MODEL, RAG_ENDPOINT, VS_INDEX
from agent import AGENT, LLM_ENDPOINT, EMBED_ENDPOINT
from mlflow.types.responses import ResponsesAgentRequest

req  = ResponsesAgentRequest(input=[{"role": "user",
    "content": "When can a glass-only auto claim be settled without an inspection?"}])
resp = AGENT.predict(req)

# ── Surface the Vector Search call ─────────────────────────────────────
rv = getattr(AGENT, "_last_retrieve", None)
if rv:
    rows = rv["rows"]
    print("─" * 64)
    print("Vector Search call")
    print("─" * 64)
    print(f"  index  : {VS_INDEX}")
    print(f"  query  : {rv['query']!r}")
    print(f"  k      : {rv['k']}")
    print(f"  results: {len(rows)} chunk(s)")
    for i, r in enumerate(rows):
        print(f"\n  [{i+1}] doc   = {r[0]}")
        print(f"       title = {r[1]}")
        print("       chunk = " + textwrap.shorten(str(r[2]), width=100, placeholder="..."))
else:
    print("(no retrieve recorded — re-run cell 2 to reload agent.py)")

# ── Agent response ───────────────────────────────────────────────
content = resp.output[0].content
text    = content[0]["text"] if isinstance(content, list) else content
print("\n" + "─" * 64)
print("Agent response")
print("─" * 64)
print(text)

# COMMAND ----------

# DBTITLE 1,Pre-deployment validation (VS connectivity + citations)
# ── Pre-deployment validation ────────────────────────────────────────────
# Both passes must be green before you run the log / deploy cells.
#
# Pass 1 — hits Vector Search directly (bypasses the agent) so you can
#          confirm the index is reachable and returning relevant chunks.
# Pass 2 — runs the full agent and checks that:
#          (a) a non-empty answer comes back, and
#          (b) at least one [source.md] citation appears in the response.
import re, textwrap
from mlflow.types.responses import ResponsesAgentRequest

TEST_QUERY = "When can a glass-only auto claim be settled without an inspection?"

# ── Pass 1: direct VS connectivity ───────────────────────────────────────
print("Pass 1 — Direct Vector Search check")
print("─" * 60)
raw  = AGENT.index.similarity_search(
    query_text=TEST_QUERY,
    columns=["doc", "title", "chunk"],
    num_results=3,
)
rows = raw.get("result", {}).get("data_array", []) or []
assert rows, "FAIL: VS returned 0 rows — index empty or endpoint unreachable"
print(f"PASS: {len(rows)} chunks retrieved")
for i, r in enumerate(rows):
    print(f"\n  [{i+1}] {r[0]}  (title: {r[1]})")
    print("       " + textwrap.shorten(r[2], width=110, placeholder="..."))

# ── Pass 2: agent end-to-end (response must cite sources) ─────────────────
print("\nPass 2 — Agent end-to-end")
print("─" * 60)
resp   = AGENT.predict(ResponsesAgentRequest(
    input=[{"role": "user", "content": TEST_QUERY}]
))
# content can be a plain str or a list of content-part dicts depending on the model
_raw   = resp.output[0].content
answer = _raw[0]["text"] if isinstance(_raw, list) else _raw
cites  = re.findall(r'\[([^\[\]]+\.(?:md|pdf|txt|docx?))\]', answer, re.IGNORECASE)
unique = list(dict.fromkeys(cites))

print("PASS: response received" if answer else "FAIL: empty response — check agent.py")
if unique:
    print(f"PASS: citations present: {unique}")
else:
    print("WARN: no [doc.md] citations found — check SYSTEM_PROMPT or sources footer in agent.py")
print("\n" + "─" * 60)
print(answer)
print("─" * 60)
print("\n✅ Ready to log and deploy." if (rows and answer) else "\n❌ Fix issues above before deploying.")

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

# DBTITLE 1,Load from UC + predict (local, no endpoint needed)
import mlflow

# UC doesn't support /latest — use the URI returned by the log cell above.
# The @mlflow.trace(span_type=RETRIEVER) on _retrieve() and langchain autolog
# capture the full call chain; open the trace link below to inspect the VS call.
loaded = mlflow.pyfunc.load_model(info.model_uri)

loaded.predict({"input": [{"role": "user",
    "content": "A visitor slipped on icy steps and went to hospital. How should this be triaged?"}]})
