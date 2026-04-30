# MLflow 3 GenAI Evaluation Walkthrough

End-to-end, runnable tour of the Databricks MLflow 3 evaluation lifecycle using a
small LangGraph ReAct agent against the public arXiv API.

If you've opened the MLflow 3 UI on Databricks lately, you'll see the GenAI panel is
organized as a single linear workflow:

> **Trace → Sessions → Judges → Evaluation Datasets → Evaluation Runs → Labeling Schemas → Labeling Sessions → Prompts / Agent Versioning → Prompt Optimization → Production Monitoring**

The one thing to internalize: **that ordering is the recommended build order.** Do
them in sequence, don't skip ahead. The companion notebook
([`arxiv_eval_walkthrough.ipynb`](arxiv_eval_walkthrough.ipynb)) runs the whole
flow end-to-end. It ships a weak v1 on purpose so you can watch a custom scorer
catch a hallucinated citation on a specific tool span, collect labels on the
failures, then fix the behaviour by versioning the prompt — no code change.

## The flow

**1. Trace.** Start every GenAI project by instrumenting. Use framework autologging
(`mlflow.langchain.autolog()`, `mlflow.openai.autolog()`) plus `@mlflow.trace` on
any tool function so tool invocations get their own named `SpanType.TOOL` spans.
Do this on day one — every stage below reads from traces.
Docs: [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)

**2. Sessions.** For multi-turn apps, tag each trace with
`mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})`. The
UI groups by this automatically. Gotcha: session metadata is immutable, so set it
before the first tool call on each turn.
Docs: [Track users and sessions](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/track-users-sessions/)

**3. Judges.** Start with the built-ins: `RelevanceToQuery`,
`RetrievalGroundedness`, `Safety`, `Correctness`, `Guidelines`. Add a custom
`@scorer` only when you need to inspect something the built-ins can't see —
typically a specific tool span's inputs or outputs. Custom code scorers are
deterministic and cheap; reach for them before you reach for another LLM judge.
The demo ships three tool-span scorers: `citation_correctness` (deterministic),
`num_results_returned` (quantitative), and `query_abstract_bleu` (BLEU-1).
Docs: [Scorers](https://mlflow.org/docs/latest/genai/concepts/scorers/)

**4. Evaluation Datasets.** Curate from real traces plus reviewer expectations,
not from synthetic prompts. Create a Unity-Catalog-backed dataset with
`mlflow.genai.create_dataset()` and grow it one production failure at a time.
Every row you add is a regression test you'll never have to write again.
Expectations are *ground truth* (e.g. `expected_arxiv_ids`); rules that apply to
every row belong in a `Guidelines` scorer, not in expectations.
Docs: [Evaluation datasets](https://mlflow.org/docs/latest/genai/concepts/evaluation-datasets/)

**5. Evaluation Runs.** `mlflow.genai.evaluate()` with a `predict_fn`, the dataset,
and your scorer list. One run per app version. The UI's side-by-side compare view
is where prompt and code changes earn their keep. If a change doesn't move a
metric you care about, it wasn't an improvement.
Docs: [Evaluate and monitor](https://mlflow.org/docs/latest/genai/eval-monitor/)

**6. Labeling Schemas.** Define before you open the session. Pick
`type="feedback"` when you want reviewer opinions, `type="expectation"` when you
want ground truth that syncs back to the dataset. A missing schema late in a
review cycle means throwing labels away.
Docs: [Labeling schemas](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-schemas)

**7. Labeling Sessions.** Route *failing* traces from an eval run into a
labeling session. Don't bulk-label; you'll burn reviewer time on examples that
were already right. Sync expectations from the session back into the evaluation
dataset so the next evaluation run uses the new ground truth.
Docs: [Labeling sessions](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-sessions)

**8. Prompts / Agent Versioning.** Every prompt lives in the UC Prompt Registry.
Use aliases (`@production`, `@candidate`) and load by alias in your agent —
promoting is a one-line alias move, no deploy. Register the agent itself with
`mlflow.set_active_model(...)` + `mlflow.log_model_params(...)`; traces link to
the active LoggedModel automatically. Bump the version every time code changes.
Never edit in place.
Docs: [Prompt registry](https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/),
[LoggedModel](https://docs.databricks.com/aws/en/mlflow3/genai/app-version/)

**9. Prompt Optimization.** `mlflow.genai.optimize_prompt(...)` samples
instruction variants against your dataset + scorers and returns the one that
maximizes the composite score. The expectations and scorers you already built
*are* the optimization spec — no extra labeling needed. Bound it with
`num_instruction_candidates` and a wall-clock timeout; optimizers can burn a lot
of LLM calls if left uncapped.
Docs: [optimize_prompt](https://mlflow.org/docs/latest/genai/prompt-optimization/)

**10. Production Monitoring.** The scorers you built for offline evaluation are
the scorers you want running in prod. `scorer.register()` attaches to the
experiment; `.start(sampling_config=ScorerSamplingConfig(sample_rate=1.0))` runs
it on every new production trace. One scorer definition, two lifecycles — dev
and prod.
Docs: [Production monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring)

## Three rules worth keeping

- Don't ship a custom LLM judge you haven't aligned on at least ten labeled
  examples.
- Every prompt change gets a prompt version; every code change gets a
  LoggedModel version. Never edit in place.
- Trace in dev; monitor the same scorers in prod. There should not be a second
  eval system.

## Files

- [`arxiv_eval_walkthrough.ipynb`](arxiv_eval_walkthrough.ipynb) — the notebook,
  run top to bottom
- [`rest_api_walkthrough.ipynb`](rest_api_walkthrough.ipynb) — companion notebook
  for non-Python frameworks; same experiment, same Prompt Registry, but every
  call is a plain HTTP `requests` call that ports directly to C# `HttpClient`,
  Java `OkHttp`, Go `net/http`, etc.
- [`arxiv_tools.py`](arxiv_tools.py) — `search_arxiv` and `fetch_arxiv_paper`;
  both wrapped as LangChain tools and decorated with
  `@mlflow.trace(span_type=SpanType.TOOL, ...)`
- [`arxiv_agent.py`](arxiv_agent.py) — LangGraph ReAct agent builder +
  `run_turn` session helper
- [`eval_questions.py`](eval_questions.py) — eight seed research questions with
  expected arXiv IDs and facts
- [`../resources/mlflow_job.yml`](../resources/mlflow_job.yml)
  — Databricks Asset Bundle job definition

## Prerequisites

- Databricks workspace with Foundation Model API access
- A Unity Catalog `catalog.schema` you can write to (for the Prompt Registry and
  the evaluation dataset)
- Internet egress to `export.arxiv.org` (the only external dependency; no auth)
- Databricks Serverless environment v4 (or equivalent ML Runtime)

## Running it

### Interactively
1. Open `arxiv_eval_walkthrough.ipynb` in a Databricks notebook.
2. Edit the `CATALOG` and `SCHEMA` cell (Step 0) to a UC location you can write
   to.
3. Run all cells. Expected runtime: 3–5 minutes.
4. When you hit the labeling session cell (Step 7), open the printed Review App
   URL, label a few traces, then continue.

### As a scheduled job
```bash
databricks bundle deploy -t dev
databricks bundle run -t dev mlflow_job
```

## REST API walkthrough (non-Python frameworks)

The Python SDK story above is the recommended path when you control the
runtime. When you don't — your agent runs in C#, Java, Go, Node — you still
want traces in the MLflow Traces UI and prompts loaded from the UC Prompt
Registry. [`rest_api_walkthrough.ipynb`](rest_api_walkthrough.ipynb) shows that
end-to-end over plain HTTP, against the same experiment as the SDK walkthrough.

Three HTTP surfaces it exercises (every call uses the standard
`Authorization: Bearer <token>` header):

- **`POST /api/2.0/otel/v1/traces`** — push traces via OTLP/HTTP+protobuf
  (Databricks rejects OTLP/JSON today). Requires `X-Databricks-UC-Table-Name`.
  Traces land in the same UC Delta tables the MLflow UI reads, so they show up
  alongside SDK traces.
  Docs: [Store traces in UC](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog).
- **`GET /api/2.0/mlflow/unity-catalog/prompts/{name}/versions/by-alias/{alias}`**
  — load a prompt template by alias (Beta surface). Path-style; the `{name}`
  segment is the three-tier UC name and must be URL-encoded. From
  [`UnityCatalogPromptService`](https://github.com/mlflow/mlflow/blob/master/mlflow/protos/unity_catalog_prompt_service.proto).
- **`DELETE FROM <prefix>_otel_*`** — UC trace tables are Delta tables; cleanup
  is plain SQL. `DeleteTracesV3` exists at
  `POST /api/2.0/mlflow/traces/delete-traces` but only operates on the legacy
  trace store, not UC tables. From a non-Python service use the
  [SQL Statement Execution API](https://docs.databricks.com/api/workspace/statementexecution)
  pointed at a SQL warehouse.

Step 1 binds the experiment to a UC trace location with one Python SDK call;
that's the only SDK code in the notebook and it's a one-time admin step. Every
other step is `requests`-only and structured for direct port to any HTTP
client. Run `arxiv_eval_walkthrough.ipynb` first so the prompt + alias exist.

## What to look for in the UI

- **Traces tab** — each agent turn has an `AGENT` span with `LLM` and named TOOL
  children (`search_arxiv`, `fetch_arxiv_paper`).
- **Sessions tab** — the two-turn demo in Step 2 shows both turns grouped.
- **Models tab** — three LoggedModel rows (`arxiv-agent-v1`, `arxiv-agent-v2`,
  optionally `arxiv-agent-v3-optimized`), each with its prompt FQN, version, and
  LLM endpoint as params.
- **Evaluation tab** — `eval-v1` / `eval-v2` (and `eval-v3`) runs side by side;
  `citation_correctness` improves from v1 to v2.
- **Prompts tab** — `{catalog}.{schema}.arxiv_agent_system_prompt` with two (or
  three) versions and two aliases.

Pick your app. Start with traces. Grow outward.
