# A production-shaped agent — governed UC tools + RAG, served on Databricks Apps

A small, self-contained reference agent that shows the **current** way to build and ship a
custom agent on Databricks. The agent answers questions about simulation runs by calling
**governed Unity Catalog SQL functions** and searching **methodology docs** — the tools are
reached through **managed MCP servers**, and the whole thing is served **in-process on a
Databricks App** (no always-on Model Serving endpoint).

It is deliberately generic and runs entirely on **seeded synthetic data** — safe to read,
fork, and hand to anyone.

| | What it shows | Key API |
|---|---|---|
| **a** | Custom agent as an MLflow **ResponsesAgent** | `ResponsesAgent` + `predict_stream` |
| **b** | **Governed SQL functions as tools**, returning compact JSON | `CREATE FUNCTION … RETURNS STRING` + Functions MCP |
| **c** | **RAG** over methodology docs | Vector Search Delta-Sync + Vector Search MCP |
| **d** | **Served on an App**, in-process | FastAPI + `from agent import AGENT` |
| **e** | **Evaluated** before shipping | `mlflow.genai.evaluate` + LLM-judge scorers |
| **f** | **Traced & attributable** per end user | `mlflow.openai.autolog()` + `mlflow.update_current_trace` |

## Use case

A team of modellers converses with a set of simulation runs: *"which models are available?"*,
*"what runs exist and their status?"*, *"for run_0001, how does output respond under
demand_up?"*. The LLM decides when to call a tool; the agent runs it, shows each step, and
answers from what came back — citing methodology docs and refusing to invent run_ids,
scenarios, metrics, or values.

## Why an App, not a Model Serving endpoint

The agent is cheap glue around two things that already scale on the platform: pay-per-token
**Foundation Model APIs** for the LLM and **managed MCP servers** for the tools. Running it
inside an App means **no dedicated serving endpoint to keep warm** (lower cost, no cold-start
hop) and the App's chat UI streams the agent's thinking directly. If you *do* need a governed
REST model (e.g. to call from a Job or another service), register and `agents.deploy` the same
`agent.py` — see [`../agents/02_custom_rag_agent.py`](../agents/02_custom_rag_agent.py).

## Files

| File | Purpose |
|------|---------|
| [`agent.py`](agent.py) | The `ResponsesAgent`: tool loop over the Functions + Vector Search MCP servers. Standalone so the App imports it and `log_model` could package it. |
| [`config.py`](config.py) | Single source of truth — catalog/schema, model + embedding endpoints, VS endpoint, tool/model names. |
| [`sql/`](sql/) | The three UC SQL tools: `read_runs`, `read_sim` (discovery → retrieval), `discover_models`. Each returns one compact JSON object. |
| [`00_setup.py`](00_setup.py) | Builds seeded synthetic tables, the Vector Search index, and registers the SQL functions. |
| [`01_evaluate.py`](01_evaluate.py) | `mlflow.genai.evaluate` quality gate (RelevanceToQuery, Safety, a grounding Guidelines judge) + seeds a few traces. |
| [`app/`](app/) | The Databricks App — FastAPI SSE server + a vanilla-JS chat UI that renders each tool step. Self-contained (its own copy of `agent.py`/`config.py`). |
| [`databricks.yml`](databricks.yml) | Folder-scoped Asset Bundle: the setup/eval job + the App. |
| [`tests/test_agent.py`](tests/test_agent.py) | Pure-function tests (no workspace needed). |
| [`MIGRATION.md`](MIGRATION.md) | Short guide for moving a "deploy-everything" notebook agent to this shape. |

## Run it

1. Edit [`config.py`](config.py) — point `CATALOG`/`SCHEMA` at a location you can write to.
2. `databricks bundle validate --strict -t dev --profile <PROFILE>` then
   `databricks bundle deploy -t dev --profile <PROFILE>`.
3. `databricks bundle run scenario_setup -t dev --profile <PROFILE>` — builds the data +
   Vector Search index, registers the tools, and runs the eval gate. (~a few minutes; the VS
   endpoint can take longer to provision the first time.)
4. The App deploys with the bundle; `databricks apps list` prints its URL. Grant the App's
   service principal: **CAN QUERY** on the LLM + embedding endpoints (declared in the bundle),
   and **USE CATALOG / USE SCHEMA / EXECUTE** on the UC functions plus **USE** on the Vector
   Search index.

Run the tests locally with no workspace: `cd scenario_agent && python -m pytest tests/ -q`.

### Prerequisites

Managed MCP must be available on the workspace (Functions + Vector Search MCP servers). If it
isn't, swap the two MCP paths in `agent.py` for `databricks_openai.UCFunctionToolkit` (UC
functions) and `VectorSearchClient().similarity_search` (RAG) — same tools, no MCP.

## What this fixes vs. a typical first-draft notebook

This is the cleaned-up shape of a common pattern. Two things first drafts get wrong, fixed
here and covered by tests:

- **No per-request state on the agent instance.** User/session identity is read from
  `custom_inputs` and written straight to the trace in `predict`/`predict_stream` — never
  cached on `self`, which leaks identity across concurrent requests.
- **Truncation never breaks JSON.** Oversized tool output is wrapped in a valid JSON envelope
  (`cap_tool_output`) rather than sliced mid-structure and handed to the model as broken JSON.

It also keeps the agent's runtime dependencies pruned to what it imports (no
langchain/langgraph), loads tools via managed MCP instead of hand-wired toolkits, and gates
deploys on `mlflow.genai.evaluate`.

## Notes

- **Tool contract.** `read_sim` has a discovery mode (call without `metrics` → metadata: the
  scenarios/metrics/steps that exist) and a retrieval mode (call with exact `metrics` →
  `data[scenario][metric]` aligned to a shared `steps` array). The system prompt tells the
  model to discover first, which keeps it from inventing names.
- **Synthetic data.** `run_0004` is intentionally left "running" with no output so you can see
  the agent report "nothing returned" instead of fabricating.
- Related: [`../tracing/`](../tracing/README.md) is the tracing/governance deep-dive;
  [`../agents/`](../agents/README.md) covers Agent Bricks + a Model-Serving custom RAG agent.
  This folder is the "governed tools + RAG, served on Apps, evaluated" slice.
