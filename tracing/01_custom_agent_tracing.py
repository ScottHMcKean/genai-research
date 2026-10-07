# Databricks notebook source
# MAGIC %md
# MAGIC # Tracing a custom agent, end to end — Monte Carlo forecasting chat
# MAGIC
# MAGIC A team of forecasters converses with a Monte Carlo simulation. The agent
# MAGIC (`agent.py`) is a small MLflow **ResponsesAgent**: the LLM decides when to run a
# MAGIC simulation, we execute it locally, and the model explains the result. The point of
# MAGIC this notebook is the **observability**, not the model. It walks the four things you
# MAGIC need to make tracing real on Databricks *today*:
# MAGIC
# MAGIC | | What | How |
# MAGIC |---|---|---|
# MAGIC | **a** | Framework autolog + a tool span | `mlflow.openai.autolog()` + `@mlflow.trace` on the simulation |
# MAGIC | **b** | Traces are OpenTelemetry and land in **Unity Catalog** | bind the experiment to a UC trace location; spans become governed Delta tables |
# MAGIC | **c** | Retrieve traces as **conversation history** | `search_traces` filtered by `mlflow.trace.user` / `mlflow.trace.session` |
# MAGIC | **d** | **Governance**: a user only sees traces they made | app pins the filter to the caller **+** a trace-scoped UC **secure view** for direct SQL |
# MAGIC
# MAGIC ### Requirements
# MAGIC - MLflow 3 (`mlflow>=3.1`), `databricks-openai`, `databricks-agents`.
# MAGIC - **Store OpenTelemetry traces in Unity Catalog** enabled for the workspace
# MAGIC   (Previews page) — needed for section **b** onward.
# MAGIC - A UC `catalog.schema` you can create tables in (see `config.py`).

# COMMAND ----------

# MAGIC %pip install --quiet -U "mlflow>=3.1" databricks-openai "databricks-sdk[openai]" databricks-agents numpy
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Config
import os
from config import (
    CATALOG, SCHEMA, CHAT_MODEL, EXPERIMENT_NAME,
    TRACE_TABLE_PREFIX, TRACE_SPANS_TABLE, TRACE_ADMIN_GROUP, SQL_WAREHOUSE_ID,
)

# MLflow needs a SQL warehouse to write/read UC-backed traces. Set it before binding.
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID

# Expects {CATALOG}.{SCHEMA} to already exist and be writable (create it once, out of
# band, if not — many governed workspaces don't grant CREATE CATALOG to end users).
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"catalog.schema : {CATALOG}.{SCHEMA}")
print(f"model          : {CHAT_MODEL}")
print(f"trace tables   : {CATALOG}.{SCHEMA}.{TRACE_TABLE_PREFIX}_otel_*")
print(f"sql warehouse  : {SQL_WAREHOUSE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## a · Framework autolog + a manual tool span
# MAGIC
# MAGIC `agent.py` calls `mlflow.openai.autolog()` at import — every LLM call is traced with
# MAGIC no code in the agent. The simulation is a plain Python function decorated with
# MAGIC `@mlflow.trace(span_type=TOOL)`, so it gets its own named span carrying the exact
# MAGIC parameters the model chose. MLflow auto-traces the ResponsesAgent's `predict` as the
# MAGIC root span, so **one turn is one trace** with the shape:
# MAGIC
# MAGIC ```
# MAGIC predict (root)                     ← auto-traced by MLflow for the ResponsesAgent
# MAGIC  ├── Completions (LLM)             ← from mlflow.openai.autolog()
# MAGIC  ├── monte_carlo_simulation (TOOL) ← from @mlflow.trace
# MAGIC  └── Completions (LLM)             ← the model explains the numbers
# MAGIC ```

# COMMAND ----------

import mlflow
from agent import AGENT
from mlflow.types.responses import ResponsesAgentRequest

user_email = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_PATH = f"/Users/{user_email}/{EXPERIMENT_NAME}"

resp = AGENT.predict(ResponsesAgentRequest(
    input=[{"role": "user", "content": (
        "Revenue is $4.2M this quarter. If it grows about 3% a quarter with a lot of "
        "uncertainty (std ~6%), what's the range in 8 quarters, and what's the chance "
        "we clear $6M?")}],
))
for it in resp.model_dump(exclude_none=True)["output"]:
    t = it.get("type")
    if t == "function_call":
        print(f"  → tool call: {it['name']}({it['arguments']})")
    elif t == "function_call_output":
        print(f"  ← tool result: {it['output']}")
    elif t == "message":
        c = it["content"]
        print("\nAnswer:\n" + (c[0]["text"] if isinstance(c, list) else c))

# COMMAND ----------

# DBTITLE 1,Inspect the spans of that trace
trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
print(f"trace_id: {trace.info.trace_id}\n")
for span in trace.data.spans:
    print(f"  {span.name:28s} {str(span.span_type):10s}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## b · Traces are OpenTelemetry, stored in Unity Catalog
# MAGIC
# MAGIC MLflow tracing on Databricks emits **OpenTelemetry** spans. Binding the experiment to
# MAGIC a UC *trace location* routes them into **Delta tables** under your `catalog.schema`
# MAGIC (`{prefix}_otel_spans`, `_otel_logs`, ...). From then on traces are governed exactly
# MAGIC like any other UC table — grants, lineage, row filters, column masks.
# MAGIC
# MAGIC > **One-shot caveat.** A UC trace location can only be attached to an experiment that
# MAGIC > has **no traces yet**. If you already ran section **a** against this experiment,
# MAGIC > either use a fresh `EXPERIMENT_NAME` or set the location from the UI
# MAGIC > (Experiment → Traces → trace location). The cell handles the re-run case cleanly.

# COMMAND ----------

from mlflow.exceptions import RestException

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

try:
    from mlflow.entities.trace_location import UnityCatalog

    mlflow.set_experiment(
        EXPERIMENT_PATH,
        trace_location=UnityCatalog(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            table_prefix=TRACE_TABLE_PREFIX,
        ),
    )
    print(f"Bound {EXPERIMENT_PATH}\n   → {CATALOG}.{SCHEMA}.{TRACE_TABLE_PREFIX}_otel_*")
except RestException as e:
    if "already" in str(e).lower():
        mlflow.set_experiment(EXPERIMENT_PATH)
        print("Experiment already has a UC trace location — continuing.")
    else:
        raise
except (TypeError, ImportError) as e:
    mlflow.set_experiment(EXPERIMENT_PATH)
    print(f"Could not bind UC location ({type(e).__name__}: {e}). "
          "Set it from the Experiment UI (Traces → trace location).")

EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id
print(f"experiment_id: {EXPERIMENT_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-user, multi-session conversation
# MAGIC
# MAGIC Three forecasters, each in their own session, some multi-turn. The agent stamps
# MAGIC `mlflow.trace.user` and `mlflow.trace.session` from `custom_inputs` (see
# MAGIC `agent.py::_tag_trace`). We deliberately vary the users so sections **c** and **d**
# MAGIC have something to separate.

# COMMAND ----------

import uuid

from mlflow.entities import SpanType

def ask(question: str, user: str, session_id: str) -> str:
    """One conversational turn, attributed to a user + session on the trace.

    We open an explicit root span for the turn (rather than relying on the agent's
    auto-traced `predict`) for two reasons: it gives a single stable, synchronous
    trace boundary — important when many turns fire back to back — and it's where we
    stamp the user + session so the metadata is set before any child span. This is the
    same call-site pattern as ../agents/05 (Probe I) and ../mlflow/arxiv_agent.run_turn."""
    with mlflow.start_span(name="forecast_turn", span_type=SpanType.AGENT) as span:
        span.set_inputs({"question": question, "user": user})
        mlflow.update_current_trace(
            metadata={"mlflow.trace.user": user, "mlflow.trace.session": session_id})
        r = AGENT.predict(ResponsesAgentRequest(
            input=[{"role": "user", "content": question}],
            custom_inputs={"user": user, "session_id": session_id},
        ))
        msgs = [o for o in r.model_dump(exclude_none=True)["output"] if o.get("type") == "message"]
        answer = (msgs[-1]["content"][0]["text"] if msgs and isinstance(msgs[-1]["content"], list)
                  else (msgs[-1]["content"] if msgs else ""))
        span.set_outputs({"answer": answer})
    return answer

# Session per (user, topic). In the app these ids come from the login + chat thread.
alice, bob = "alice@example.com", "bob@example.com"
s_alice = f"sess-{uuid.uuid4().hex[:8]}"
s_bob   = f"sess-{uuid.uuid4().hex[:8]}"

conversation = [
    ("Project $4.2M revenue over 8 quarters at ~3% growth, std 6%. Give me the range.", alice, s_alice),
    ("Now what's the probability we clear $6M by then?",                                  alice, s_alice),
    ("And the downside — what's the p10 if growth is only 1% with std 8%?",               alice, s_alice),
    ("Model 12,000 units growing 2% per month, std 4%, over 18 months.",                  bob,   s_bob),
    ("What are the odds we stay above 15,000 units?",                                     bob,   s_bob),
    # A third user with a single turn — used to prove isolation in section d.
    ("Simulate a $900k budget shrinking 1% a period, std 3%, over 12 periods.",
     "carol@example.com", f"sess-{uuid.uuid4().hex[:8]}"),
]
for q, u, s in conversation:
    ans = ask(q, u, s)
    print(f"[{u:18s} {s}] {q[:60]}\n    → {ans[:120]}\n")

print(f"{len(conversation)} traces written. In the Traces UI they carry user + session; "
      "the Sessions tab groups Alice's 3 turns and Bob's 2 turns.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The traces are a governed UC Delta table
# MAGIC
# MAGIC Traces stream to UC asynchronously — give it a few seconds, then query the spans
# MAGIC table directly. This is the payoff of **b**: your observability data is SQL-queryable
# MAGIC and lives under Unity Catalog governance.

# COMMAND ----------

import time

# Traces export asynchronously, so poll until they're queryable rather than guessing a
# sleep — otherwise the reads in sections c/d can race the flush and undercount.
def wait_for_traces(experiment_id: str, expected: int, timeout_s: int = 150, interval_s: int = 5) -> int:
    deadline = time.time() + timeout_s
    n = 0
    while time.time() < deadline:
        n = len(mlflow.search_traces(experiment_ids=[experiment_id], return_type="list"))
        if n >= expected:
            break
        time.sleep(interval_s)
    return n

n_ready = wait_for_traces(EXPERIMENT_ID, len(conversation))
print(f"{n_ready}/{len(conversation)} conversation traces queryable in UC\n")

# The exact column layout is versioned by the platform, so DESCRIBE before you depend
# on a column name (we key the secure view in section d off what this shows —
# the user lands as the OTel attribute `user.id` inside the `attributes` VARIANT).
print(f"Schema of {TRACE_SPANS_TABLE}:")
display(spark.sql(f"DESCRIBE {TRACE_SPANS_TABLE}"))
print("Row count:")
display(spark.sql(f"SELECT count(*) AS spans FROM {TRACE_SPANS_TABLE}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## c · Retrieve traces as conversation history
# MAGIC
# MAGIC `mlflow.search_traces` reads the same UC data back. Filter by
# MAGIC `metadata.\`mlflow.trace.user\`` and `metadata.\`mlflow.trace.session\`` to
# MAGIC reconstruct exactly one person's conversation, in order — the backbone of a
# MAGIC "your history" panel in an app.

# COMMAND ----------

import json

def _as_obj(x):
    """search_traces fields may arrive as JSON strings or already-parsed objects."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def _question_of(trace) -> str:
    # The turn's root span (opened in ask()) carries {"question", "user", ...} in its
    # inputs. Read that directly — trace.data.request reflects the inner auto-traced
    # predict span (the ResponsesAgent request shape), not our root.
    try:
        root = next((s for s in (trace.data.spans or [])
                     if getattr(s, "parent_id", None) is None), None)
        ins = getattr(root, "inputs", None) if root else None
        ins = _as_obj(ins)
        if isinstance(ins, dict):
            if ins.get("question"):
                return ins["question"]
            if isinstance(ins.get("input"), list):
                return ins["input"][-1].get("content", "")
    except Exception:
        pass
    return str(_as_obj(trace.data.request))[:120]

def _answer_of(trace) -> str:
    res = _as_obj(trace.data.response)
    try:
        out = res["output"] if isinstance(res, dict) else res
        return next((c["content"][0]["text"] for c in out if c.get("type") == "message"), "")
    except Exception:
        return str(res)[:200]

def conversation_history(experiment_id: str, user: str, session_id: str) -> list[dict]:
    """Reconstruct one user's session as an ordered list of {question, answer} turns.

    Uses return_type='list' for stable Trace objects across MLflow versions (the
    DataFrame column names changed between 2.x and 3.x)."""
    traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=(f"metadata.`mlflow.trace.user` = '{user}' AND "
                       f"metadata.`mlflow.trace.session` = '{session_id}'"),
        order_by=["timestamp_ms ASC"],
        return_type="list",
    )
    return [{"trace_id": t.info.trace_id, "question": _question_of(t), "answer": _answer_of(t)}
            for t in traces]

for turn in conversation_history(EXPERIMENT_ID, alice, s_alice):
    print(f"Q: {turn['question'][:80]}")
    print(f"A: {turn['answer'][:140]}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## d · Governance — a user only sees traces they made
# MAGIC
# MAGIC Two layers, defence in depth:
# MAGIC
# MAGIC 1. **Application layer.** The app authenticates the caller (Databricks Apps forwards
# MAGIC    the end-user's identity) and *pins* the search filter to that identity. There is no
# MAGIC    code path that returns someone else's traces — see `app/app.py::history`.
# MAGIC 2. **Data layer (Unity Catalog).** Because the traces are a UC table, we enforce
# MAGIC    per-user isolation in SQL too, so it holds even for a direct query from a notebook,
# MAGIC    DBSQL, or a BI tool. **Schema reality check:** the OTel `_otel_spans` table is
# MAGIC    *per span*, and MLflow puts the user only on each trace's **root** span (as the
# MAGIC    OTel attribute `user.id`; session is `session.id`). A naive per-row filter on
# MAGIC    `user.id` would therefore hide every child LLM/TOOL span. The correct primitive is
# MAGIC    a **trace-scoped secure view**: resolve each trace's owner from its root span, then
# MAGIC    return *all* spans of the traces `current_user()` owns. Grant users the view, not
# MAGIC    the base table. Members of `TRACE_ADMIN_GROUP` see everything.

# COMMAND ----------

# DBTITLE 1,Application-layer filter — the only query the app ever runs
def my_history(experiment_id: str, caller: str):
    """What the app calls. `caller` comes from the authenticated request, never the
    client body — so a user can only ever ask for their own traces."""
    return mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=f"metadata.`mlflow.trace.user` = '{caller}'",
        order_by=["timestamp_ms DESC"],
        return_type="list",
    )

n_alice, n_bob = len(my_history(EXPERIMENT_ID, alice)), len(my_history(EXPERIMENT_ID, bob))
print(f"Alice sees {n_alice} traces; Bob sees {n_bob}; "
      "neither can see the other's — the filter is fixed to the caller.")

# COMMAND ----------

# DBTITLE 1,Data-layer enforcement — a trace-scoped secure view
# The user lands on each trace's ROOT span as the OTel attribute `user.id` (session as
# `session.id`), inside the `attributes` VARIANT. Note the dotted key needs bracket
# syntax: attributes:['user.id']. The view resolves each trace's owner from its root
# span, then returns every span of the traces the caller owns.
MY_SPANS_VIEW = f"{CATALOG}.{SCHEMA}.{TRACE_TABLE_PREFIX}_my_spans"

spark.sql(f"""
CREATE OR REPLACE VIEW {MY_SPANS_VIEW} AS
WITH trace_owner AS (
  SELECT trace_id, max(attributes:['user.id']::string) AS trace_user
  FROM {TRACE_SPANS_TABLE}
  GROUP BY trace_id
)
SELECT s.*, o.trace_user
FROM {TRACE_SPANS_TABLE} s
JOIN trace_owner o ON s.trace_id = o.trace_id
WHERE is_account_group_member('{TRACE_ADMIN_GROUP}')   -- auditors see everything
   OR o.trace_user = current_user()                     -- everyone else: only their own
""")
print(f"Created secure view {MY_SPANS_VIEW}.")
print("Grant forecasters the VIEW (not the base table) so direct SQL is scoped too:")
print(f"  GRANT SELECT ON VIEW {MY_SPANS_VIEW} TO `forecasters`")

# Prove the scoping logic: how many spans each user would see through the view.
owners = spark.sql(f"""
WITH trace_owner AS (
  SELECT trace_id, max(attributes:['user.id']::string) AS trace_user
  FROM {TRACE_SPANS_TABLE} GROUP BY trace_id)
SELECT o.trace_user,
       count(DISTINCT s.trace_id) AS traces_visible,
       count(*) AS spans_visible
FROM {TRACE_SPANS_TABLE} s JOIN trace_owner o ON s.trace_id = o.trace_id
WHERE o.trace_user IS NOT NULL
GROUP BY o.trace_user ORDER BY o.trace_user
""")
print("\nPer-user visibility through the view (whole traces, root + child spans):")
display(owners)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verifying the view
# MAGIC
# MAGIC The view evaluates `current_user()` as the *querying* identity, so to prove isolation
# MAGIC have two non-admin forecasters each `SELECT count(*) FROM <..._my_spans>` — each sees
# MAGIC only the spans of traces they own, and neither sees the other's. From your own
# MAGIC (non-owner, non-admin) session the view returns **0** rows, which is itself the proof.
# MAGIC The `display(owners)` above shows the per-user counts the view enforces.
# MAGIC
# MAGIC For a hard guarantee, revoke `SELECT` on the base `_otel_spans` table from forecasters
# MAGIC and grant only the view — then the base rows are unreachable except through the
# MAGIC scoped view (and to `TRACE_ADMIN_GROUP`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## e · Latency profile
# MAGIC
# MAGIC Two questions in production: how long does a **turn** take (generation), and how long
# MAGIC does it take to **read traces back** (retrieval, e.g. loading a user's history in the
# MAGIC app)? Both come straight from what we already captured — no extra instrumentation.

# COMMAND ----------

# DBTITLE 1,Generation latency — where the time in a turn goes (from span durations in UC)
display(spark.sql(f"""
SELECT coalesce(attributes:['mlflow.spanType']::string, name) AS span_type,
       count(*)                                                        AS n,
       round(percentile((end_time_unix_nano-start_time_unix_nano)/1e6, 0.5), 0)  AS p50_ms,
       round(percentile((end_time_unix_nano-start_time_unix_nano)/1e6, 0.95), 0) AS p95_ms,
       round(max((end_time_unix_nano-start_time_unix_nano)/1e6), 0)              AS max_ms
FROM {TRACE_SPANS_TABLE}
GROUP BY 1
ORDER BY p50_ms DESC
"""))
# The Monte Carlo tool span is ~10 ms; essentially all turn latency is the LLM. A
# tool-using turn makes two LLM calls (decide to call the tool, then explain the result).

# COMMAND ----------

# DBTITLE 1,Retrieval latency — reading traces back from the tracking server
import statistics

def _timeit(fn, n: int = 6) -> dict:
    """Cold (first call) vs warm (median of the rest) round-trip, in ms."""
    ts = []
    for _ in range(n):
        t = time.time(); fn(); ts.append((time.time() - t) * 1000)
    return {"cold_ms": round(ts[0]), "warm_p50_ms": round(statistics.median(ts[1:]))}

a_traces = mlflow.search_traces(
    experiment_ids=[EXPERIMENT_ID],
    filter_string=f"metadata.`mlflow.trace.user` = '{alice}'", return_type="list")
a_tid = a_traces[0].info.trace_id

retrieval = {
    "search_all":      _timeit(lambda: mlflow.search_traces(experiment_ids=[EXPERIMENT_ID], return_type="list")),
    "search_by_user":  _timeit(lambda: mlflow.search_traces(experiment_ids=[EXPERIMENT_ID],
                               filter_string=f"metadata.`mlflow.trace.user` = '{alice}'", return_type="list")),
    "get_trace_by_id": _timeit(lambda: mlflow.get_trace(a_tid)),
}
for k, v in retrieval.items():
    print(f"  {k:18s} cold={v['cold_ms']:>5} ms   warm_p50={v['warm_p50_ms']:>5} ms")
# search_traces goes through the SQL warehouse (~3 s warm here, ~roughly flat across
# filters at this data size; cold adds warehouse/query-plan startup). get_trace by id is
# a direct fetch (~0.6 s warm). For the app: load history once on open; fetch a single
# trace on click. Neither is on the hot path of a chat turn.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC - **a** — `mlflow.openai.autolog()` + `@mlflow.trace` gave every turn a clean span tree
# MAGIC   with the simulation as its own tool span. No trace code in the agent loop itself.
# MAGIC - **b** — one `set_experiment(trace_location=UnityCatalog(...))` call put the OTel
# MAGIC   spans into governed Delta tables you can query with SQL.
# MAGIC - **c** — `search_traces` filtered by user + session reconstructs a conversation; that
# MAGIC   *is* your chat history store, no separate database.
# MAGIC - **d** — the app pins the filter to the caller **and** a trace-scoped UC secure view
# MAGIC   enforces per-user isolation at the data layer, so governance doesn't depend on app code.
# MAGIC
# MAGIC Next: `app/` deploys this same agent as a Databricks App where each forecaster chats
# MAGIC and sees only their own history — governance **c** + **d** made concrete.

# COMMAND ----------

# DBTITLE 1,End-to-end summary (returned by the job run)
alice_turns = conversation_history(EXPERIMENT_ID, alice, s_alice)
spans_in_uc = spark.sql(f"SELECT count(*) c FROM {TRACE_SPANS_TABLE}").first()["c"]

summary = {
    "experiment_id": EXPERIMENT_ID,
    "a_spans_in_last_trace": len(trace.data.spans),
    "b_spans_rows_in_uc": int(spans_in_uc),
    "c_alice_session_turns": len(alice_turns),
    "d_alice_visible": n_alice,
    "d_bob_visible": n_bob,
    "trace_table": TRACE_SPANS_TABLE,
}
print(summary)

# Return the summary so a headless job run surfaces the result.
try:
    dbutils.notebook.exit(json.dumps(summary))
except Exception:
    pass
