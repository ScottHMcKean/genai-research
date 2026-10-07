# Tracing a custom agent, end to end (MLflow 3 on Databricks)

A clear, current example of **observability for a custom agent** — not a framework
demo, a *tracing* demo. The agent is deliberately small so the four things that
actually matter in production stay in focus:

| | What it shows | Key API |
|---|---|---|
| **a** | Framework **autolog** + a manual tool span | `mlflow.openai.autolog()` + `@mlflow.trace(span_type=TOOL)` |
| **b** | Traces are **OpenTelemetry** and land in **Unity Catalog** | `mlflow.set_experiment(trace_location=UnityCatalog(...))` |
| **c** | Retrieve traces as **conversation history** (users + sessions) | `mlflow.search_traces(filter_string="metadata.\`mlflow.trace.user\` = …")` |
| **d** | **Governance** — a user only sees traces they made | app pins the filter to the caller **+** a trace-scoped UC **secure view** for direct SQL |

## Use case

A team of forecasters converses with a Monte Carlo simulation. The LLM decides when
to run a projection, the agent runs it (`run_monte_carlo`, seeded NumPy), and the
model explains the p10/p50/p90 range and any target probability in plain language.
Every turn is one trace; every trace is attributable to a person.

## Files

- [`agent.py`](agent.py) — `ForecastChatAgent`, an MLflow **ResponsesAgent**. LLM tool
  loop + the `@mlflow.trace`-decorated `run_monte_carlo` tool. `predict` is the root
  `AGENT` span and stamps `mlflow.trace.user` / `mlflow.trace.session` from
  `custom_inputs`. Standalone so it can be logged and imported by the app.
- [`01_custom_agent_tracing.py`](01_custom_agent_tracing.py) — the walkthrough
  notebook. Sections **a → d**, runnable top to bottom on serverless.
- [`config.py`](config.py) — catalog/schema, model, trace-table prefix, admin group.
- [`app/`](app/README.md) — the same agent as a **Databricks App**: forecasters chat
  and each sees only their own history. Makes **c** + **d** concrete.
- [`WIRING.md`](WIRING.md) — how it's wired: components, the exact queries behind each
  piece, measured latencies, permissions checklist, and doc references.

## The one thing to internalise

On Databricks, MLflow traces **are** OpenTelemetry, and once an experiment is bound
to a UC trace location they **are** governed Delta tables. So you don't build a
separate chat-history database and you don't build a separate access-control layer:

- **History** is a `search_traces` filtered by user + session.
- **Isolation** has two layers. In the app it's enforced at the **app layer** — every
  history query is pinned to the authenticated caller. For direct SQL / BI access it's a
  trace-scoped **Unity Catalog secure view** (`…_my_spans`): resolve each trace's owner
  from its root span, return only the traces `current_user()` owns. Grant users the
  **view**, not the base table — the base `_otel_spans` table is not itself row-filtered
  (a per-row filter can't scope by trace, since the user is only on the root span).

## Run it

1. Enable **Store OpenTelemetry traces in Unity Catalog** on the workspace Previews
   page.
2. Edit `config.py` (`CATALOG`, `SCHEMA`) to a location you can create tables in.
3. Open `01_custom_agent_tracing.py` on serverless and run all. ~3 minutes.
4. Optional — deploy the app: see [`app/README.md`](app/README.md).

### What to look at afterwards

- **Traces tab** — each `forecast_turn` has an `LLM` span, a `monte_carlo_simulation`
  `TOOL` span, and a second `LLM` span, tagged with user + session.
- **Sessions tab** — Alice's 3 turns and Bob's 2 turns each group into one session.
- **Catalog** — `{catalog}.{schema}.forecast_traces_otel_spans` is a queryable, governed
  Delta table; `…_my_spans` is the trace-scoped secure view over it.

## Notes

- **The user is on the root span only.** MLflow exports `mlflow.trace.user` to the OTel
  attribute `user.id` (session → `session.id`) inside the `attributes` VARIANT, and only
  on each trace's root span. The `_otel_spans` table is per-span, so the secure view
  resolves each trace's owner from its root span and returns all spans of owned traces.
  Column layout is platform-versioned — the notebook `DESCRIBE`s the table first.
- **Binding UC trace location is one-shot** — it only attaches to an experiment with no
  traces yet. Use a fresh experiment name or set it from the Experiment UI on re-runs.
- **User identity must be set explicitly.** Under app auth the workspace identity is the
  service principal, so the app forwards the *authenticated end user* into
  `mlflow.trace.user` — that's what the secure view and the history filter key on.
- Related: [`../mlflow/`](../mlflow/README.md) covers the full eval lifecycle (judges,
  datasets, prompt registry); [`../agents/`](../agents/README.md) covers Agent Bricks +
  a custom RAG agent. This folder is the tracing/governance slice.
