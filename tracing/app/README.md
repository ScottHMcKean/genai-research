# Forecast Chat — the agent as a Databricks App

Deploys the Monte Carlo forecasting agent as a chat app where each forecaster sees
**only their own** trace history. This is sections **c** + **d** of
[`../01_custom_agent_tracing.py`](../01_custom_agent_tracing.py) made concrete.

## How governance works here

- **Identity is trusted from the platform, not the client.** Databricks Apps forwards
  the signed-in user on `X-Forwarded-Email`. `app.py::caller()` reads that header; the
  request body never carries a user id.
- **Every turn is attributed.** `POST /api/chat` passes the caller as
  `custom_inputs={"user": ...}`, so the agent stamps `mlflow.trace.user` on the trace.
  Note this is the *end user*, not the app's service principal.
- **History is pinned to the caller.** `GET /api/history` runs exactly one
  `search_traces` with `filter_string` fixed to the authenticated user. There is no
  parameter a client could set to read another user's traces.
- **The data layer enforces the same rule for direct SQL.** The UC secure view from the
  notebook (`…_my_spans`) returns only the caller's own traces to `current_user()`. Grant
  analysts/BI the **view**, not the base `_otel_spans` table. (The app's own history is
  enforced at the app layer above — the view is the guarantee for direct SQL / BI access.)

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | chat UI |
| GET | `/api/me` | the authenticated user |
| POST | `/api/chat` | run one turn (JSON, non-streaming); tags the trace with user + session; returns the answer, the tool call, and the simulation chart |
| GET | `/api/history` | the caller's own past turns (each links to its session) |
| GET | `/api/session?session_id=` | reconstruct one of the caller's conversations (question, answer, chart per turn) — this is what clicking a history item loads |

> **Why non-streaming?** Server-Sent Events over the Apps proxy drop long-lived
> connections (the browser sees `TypeError: network error`). A turn is only a few
> seconds, so we run it server-side and return one JSON payload — robust for a demo.

## Artifacts (the simulation chart)

Each simulation produces a distribution chart (histogram of terminal values with
p10/p50/p90 and the target line). We persist it to a **Unity Catalog Volume**
(`/Volumes/<catalog>/<schema>/artifacts/<trace-id>.png`), tag the trace with
`artifact.chart = <path>` for lineage, and inline the PNG in the response.

**Volume vs. MLflow run artifacts — why the Volume.** For an app that *serves*
artifacts back, a Volume read is a single Files-API GET of a small file (sub-second,
and we base64-inline it). MLflow run artifacts add hops — resolve the run, then hit the
artifact store — and tie the file to a run rather than the trace the user is looking at.
Same UC governance either way; the Volume is simply the faster serving path. Access is
scoped per user: `/api/session` only returns the caller's own traces, so any chart it
inlines already belongs to them.

## Prerequisites

- Run `../01_custom_agent_tracing.py` section **b** once to bind the experiment
  (`FORECAST_EXPERIMENT_PATH`, default `/Shared/forecast_tracing`) to a UC trace
  location. The app writes to and reads from that experiment.
- Grant the app's service principal:
  - `CAN_QUERY` on the `databricks-claude-sonnet-4-5` serving endpoint,
  - `EDIT` (write traces) on the experiment,
  - `CAN_USE` on the SQL warehouse and `SELECT, MODIFY` on the `…_otel_*` trace tables
    (write + read traces via `search_traces`),
  - `READ VOLUME, WRITE VOLUME` on the artifacts volume (chart persistence).

## Run locally

```bash
cd tracing/app
DEV_USER=you@example.com FORECAST_EXPERIMENT_PATH=/Users/you@example.com/forecast_tracing \
  uvicorn app:app --reload --port 8000
```

`DEV_USER` stands in for the identity header when running outside Apps.

## Deploy as a Databricks App

```bash
databricks apps create forecast-chat
databricks sync tracing/app /Workspace/Users/<you>/forecast-chat-src
databricks apps deploy forecast-chat --source-code-path /Workspace/Users/<you>/forecast-chat-src
```

Set `FORECAST_EXPERIMENT_PATH` in `app.yaml` to the experiment you bound in the
notebook, then open the app URL and chat. Each signed-in user sees only their history.
