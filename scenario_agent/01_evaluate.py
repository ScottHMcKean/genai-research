# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Evaluate the agent (quality gate) + seed traces
# MAGIC
# MAGIC Runs the agent in-process over a small question set and scores it with
# MAGIC `mlflow.genai.evaluate` using built-in LLM-judge scorers. This is the gate you run
# MAGIC before shipping a new version of the app. It also populates a few traces so the
# MAGIC Traces tab has something to look at.
# MAGIC
# MAGIC Requires `mlflow>=3.8` for `mlflow.genai`. Run on serverless **after** `00_setup.py`.
# MAGIC
# MAGIC The `protobuf>=6.33.5` pin keeps the install from downgrading protobuf below the
# MAGIC version Spark Connect's generated code needs — without it, `restartPython` leaves the
# MAGIC serverless kernel unable to re-init Spark.

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.8" databricks-openai databricks-mcp mcp "protobuf>=6.33.5"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.types.responses import ResponsesAgentRequest

import config  # noqa: F401  (imported so the agent's config travels with code_paths)
from agent import AGENT, _text

mlflow.openai.autolog()
me = WorkspaceClient().current_user.me().user_name  # SDK, not Spark — this task needs no Spark
EXPERIMENT = f"/Users/{me}/scenario_agent_eval"
mlflow.set_experiment(EXPERIMENT)
print("Experiment:", EXPERIMENT)


def answer(question: str) -> str:
    """Call the agent for one turn and return the final text answer."""
    resp = AGENT.predict(ResponsesAgentRequest(
        input=[{"role": "user", "content": question}],
        custom_inputs={"user": "eval@demo", "session_id": "eval"},
    ))
    for item in reversed(resp.output):
        d = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
        if d.get("type") == "message":
            return _text(d.get("content"))
    return ""

# COMMAND ----------

# MAGIC %md ## Evaluation set (small, illustrative — grow from production logs in practice)

# COMMAND ----------

data = [
    {"inputs": {"question": "Which models are available to simulate against?"}},
    {"inputs": {"question": "What runs exist and what is the status of each?"}},
    {"inputs": {"question": "For run_0001, which scenarios and metrics are available?"}},
    {"inputs": {"question": "For run_0001, how does output respond under the demand_up scenario?"}},
    {"inputs": {"question": "What does the methodology say a 'step' means?"}},
]

# COMMAND ----------

# MAGIC %md ## Score with built-in judges
# MAGIC `RelevanceToQuery` and `Safety` need no ground truth. The `Guidelines` judge sees only
# MAGIC the request + final answer (not the tool outputs), so the rule is worded for what it can
# MAGIC observe: concrete identifiers drawn from the data are *expected* (not fabrication), and
# MAGIC the answer must flag missing data rather than invent values. To check grounding against
# MAGIC the retrieved *context* instead, pass it via expectations or emit RETRIEVER spans and use
# MAGIC `RetrievalGroundedness`.

# COMMAND ----------

from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety

faithful = Guidelines(
    name="faithful",
    guidelines=(
        "Evaluate the assistant's final answer only. PASS if it answers the question with "
        "concrete specifics drawn from the data (run ids like run_0001, scenario or metric "
        "names, step values, or a cited methodology file such as [overview.md]) — giving such "
        "specifics is expected and is NOT fabrication. Also PASS if it says no data was found "
        "for something that does not exist. FAIL only if it is internally contradictory, or "
        "claims to lack data while simultaneously reporting specific values for it."
    ),
)

results = mlflow.genai.evaluate(
    data=data,
    predict_fn=answer,
    scorers=[RelevanceToQuery(), Safety(), faithful],
)

print("Aggregate metrics:")
for k, v in results.metrics.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC Open the **Evaluations** tab on the experiment for per-row judge rationales, and the
# MAGIC **Traces** tab for the tool-call spans. Promote the app (`01` green → `databricks
# MAGIC bundle deploy`) once the pass rates look good.
