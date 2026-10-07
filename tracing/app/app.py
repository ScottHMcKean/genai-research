"""Databricks App — forecasting chat over the Monte Carlo agent, with per-user
trace history and governance.

What this app demonstrates (sections c + d of ../01_custom_agent_tracing.py):
- Every turn is traced; the agent stamps the trace with the **authenticated end
  user** (from the Apps identity header) and the chat session id.
- `GET /api/history` reconstructs a user's past conversations with
  `mlflow.search_traces` — and pins the filter to the caller, so a user can only
  ever retrieve their own traces. The Unity Catalog secure view in the notebook
  (`…_my_spans`) enforces the same rule at the data layer for direct SQL / BI access.

Identity: Databricks Apps forwards the signed-in user on `X-Forwarded-Email`
(and `X-Forwarded-Preferred-Username`). We trust those headers, never a user id
from the request body.
"""
import base64
import io
import json
import os
import pathlib
import re
import uuid

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import mlflow
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from mlflow.types.responses import ResponsesAgentRequest

from agent import AGENT

# Traces flow to this experiment (bound to a UC trace location by the notebook).
EXPERIMENT_PATH = os.environ.get("FORECAST_EXPERIMENT_PATH", "/Shared/forecast_tracing")
# Chart artifacts persist here (a UC Volume — faster to serve than MLflow run
# artifacts, and governed by Unity Catalog like the traces). One PNG per trace.
ARTIFACT_VOLUME = os.environ.get(
    "FORECAST_ARTIFACT_VOLUME", "/Volumes/shm_skunkworks_catalog/forecast_demo/artifacts")
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_PATH)
EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id
_ws = WorkspaceClient()

app = FastAPI(title="Forecast Chat")
_HTML = (pathlib.Path(__file__).parent / "static" / "index.html").read_text()


def make_chart_png(args: dict) -> bytes:
    """Render the Monte Carlo terminal-value distribution for one simulation. Re-runs
    the sim from the tool's own args with the same seed (42), so the chart matches the
    numbers the agent reported exactly."""
    rng = np.random.default_rng(42)
    sv, gm, gs = float(args["start_value"]), float(args["growth_mean"]), float(args["growth_std"])
    periods, n = int(args["periods"]), int(args.get("n_sims", 20000))
    finals = sv * np.prod(1.0 + rng.normal(gm, gs, size=(n, periods)), axis=1)
    p10, p50, p90 = np.percentile(finals, [10, 50, 90])

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=110)
    ax.hist(finals, bins=60, color="#ff6f00", alpha=0.85, edgecolor="none")
    top = ax.get_ylim()[1]
    for v, lbl, col in [(p10, "p10", "#888"), (p50, "p50", "#111"), (p90, "p90", "#888")]:
        ax.axvline(v, color=col, ls="--", lw=1)
        ax.text(v, top * 0.94, f" {lbl}={v:,.0f}", fontsize=8, color=col, rotation=90, va="top")
    thr = args.get("threshold")
    if thr is not None:
        ax.axvline(float(thr), color="#0a8f5b", lw=1.6)
        ax.text(float(thr), top * 0.5, f" target {float(thr):,.0f}", fontsize=8, color="#0a8f5b")
    ax.set_title(f"Monte Carlo: {sv:,.0f} → {periods} periods @ {gm:.0%} ± {gs:.0%}", fontsize=9)
    ax.set_xlabel("terminal value"); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png"); plt.close(fig)
    return buf.getvalue()


def _artifact_path(trace_id: str) -> str:
    """UC trace ids look like 'trace:/cat.schema.prefix/<hex>' — slashes and a colon.
    Slug them into a safe single filename in the Volume."""
    return f"{ARTIFACT_VOLUME}/{re.sub(r'[^A-Za-z0-9_-]', '_', trace_id)}.png"

def _read_chart_b64(trace_id: str):
    try:
        data = _ws.files.download(_artifact_path(trace_id)).contents.read()
        return "data:image/png;base64," + base64.b64encode(data).decode()
    except Exception:
        return None

def _root_span(trace):
    return next((s for s in (trace.data.spans or []) if getattr(s, "parent_id", None) is None), None)

def _as_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return x or {}

def _qa(trace):
    """(question, answer) from the root forecast_turn span's inputs/outputs."""
    r = _root_span(trace)
    ins, outs = _as_obj(getattr(r, "inputs", None)), _as_obj(getattr(r, "outputs", None))
    return ins.get("question", ""), (outs.get("answer", "") if isinstance(outs, dict) else "")


def caller(request: Request) -> str:
    """The authenticated end user, from the Apps identity headers. Falls back to a
    local dev value only when the headers are absent (running outside Apps)."""
    return (
        request.headers.get("X-Forwarded-Email")
        or request.headers.get("X-Forwarded-Preferred-Username")
        or os.environ.get("DEV_USER", "local-dev@example.com")
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/me")
def me(request: Request) -> dict:
    return {"user": caller(request)}


@app.post("/api/chat")
async def chat(request: Request) -> JSONResponse:
    """One turn, non-streaming. SSE through the Apps proxy is flaky on long-lived
    connections (browsers see a mid-stream 'network error'), so we run the whole turn
    server-side and return one JSON payload — more robust for a demo, and the turn is
    only a few seconds. The tool call is returned so the UI can still show the step."""
    body = await request.json()
    user = caller(request)                       # trusted identity, not from the body
    session_id = body.get("session_id") or f"sess-{uuid.uuid4().hex[:8]}"
    messages = body.get("input") or [{"role": "user", "content": body.get("message", "")}]
    question = messages[-1].get("content", "") if messages else ""

    try:
        # Explicit synchronous root span per turn: stable trace boundary + user/session
        # stamped before any child span. Matches ../01_custom_agent_tracing.py::ask.
        with mlflow.start_span(name="forecast_turn", span_type=SpanType.AGENT) as span:
            span.set_inputs({"question": question, "user": user, "session_id": session_id})
            mlflow.update_current_trace(
                metadata={"mlflow.trace.user": user, "mlflow.trace.session": session_id})
            resp = AGENT.predict(ResponsesAgentRequest(
                input=messages, custom_inputs={"user": user, "session_id": session_id}))
            out = resp.model_dump(exclude_none=True)["output"]
            tool_calls = [{"name": o.get("name"), "arguments": o.get("arguments"),
                           "output": None} for o in out if o.get("type") == "function_call"]
            outputs = [o.get("output") for o in out if o.get("type") == "function_call_output"]
            for i, o in enumerate(outputs):  # pair each call with its result for the UI
                if i < len(tool_calls):
                    tool_calls[i]["output"] = o
            msgs = [o for o in out if o.get("type") == "message"]
            answer = (msgs[-1]["content"][0]["text"] if msgs and isinstance(msgs[-1]["content"], list)
                      else (msgs[-1]["content"] if msgs else ""))
            span.set_outputs({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    # Artifact: render the simulation's distribution chart, persist it to the UC Volume
    # keyed by trace id (so it's retrievable from history), and inline it for the live
    # turn (instant, no dependency on the async trace flush). Never fail the turn on it.
    trace_id = mlflow.get_last_active_trace_id()
    chart_b64 = None
    if tool_calls and trace_id:
        try:
            png = make_chart_png(json.loads(tool_calls[0]["arguments"] or "{}"))
            path = _artifact_path(trace_id)
            _ws.files.upload(path, io.BytesIO(png), overwrite=True)
            try:
                mlflow.set_trace_tag(trace_id, "artifact.chart", path)  # lineage: trace → artifact
            except Exception:
                pass
            chart_b64 = "data:image/png;base64," + base64.b64encode(png).decode()
        except Exception:
            chart_b64 = None

    return JSONResponse({"answer": answer, "tool_calls": tool_calls,
                         "session_id": session_id, "trace_id": trace_id, "chart_b64": chart_b64})


@app.get("/api/session")
def session(request: Request, session_id: str) -> JSONResponse:
    """Reconstruct one of the caller's past conversations (pinned to the caller)."""
    user = caller(request)
    traces = mlflow.search_traces(
        experiment_ids=[EXPERIMENT_ID],
        filter_string=(f"metadata.`mlflow.trace.user` = '{user}' AND "
                       f"metadata.`mlflow.trace.session` = '{session_id}'"),
        order_by=["timestamp_ms ASC"], return_type="list")
    turns = []
    for t in traces:
        q, a = _qa(t)
        # Inline the chart from the Volume. Safe: this list is already pinned to the
        # caller's own traces, so a returned artifact always belongs to them.
        turns.append({"trace_id": t.info.trace_id, "question": q, "answer": a,
                      "chart_b64": _read_chart_b64(t.info.trace_id)})
    return JSONResponse({"session_id": session_id, "turns": turns})


@app.get("/api/history")
def history(request: Request) -> JSONResponse:
    """Return the caller's conversations, grouped by session (one entry per session).
    The filter is fixed to the authenticated user — there is no parameter a client
    could set to see another user's traces."""
    user = caller(request)
    # return_type='list' gives stable Trace objects across MLflow versions.
    traces = mlflow.search_traces(
        experiment_ids=[EXPERIMENT_ID],
        filter_string=f"metadata.`mlflow.trace.user` = '{user}'",
        order_by=["timestamp_ms DESC"],
        max_results=200,
        return_type="list",
    )
    # Traces arrive newest-first. Group by session: first time we see a session is its
    # latest turn (preserves ordering); we keep overwriting the title so it ends as the
    # session's earliest question (the conversation's opening line).
    order: list[str] = []
    sessions: dict[str, dict] = {}
    for t in traces:
        sid = (t.info.trace_metadata or {}).get("mlflow.trace.session") or "(no session)"
        q, _ = _qa(t)
        if sid not in sessions:
            sessions[sid] = {"session_id": sid, "turns": 0, "title": q}
            order.append(sid)
        sessions[sid]["turns"] += 1
        if q:
            sessions[sid]["title"] = q          # last write (earliest turn) wins → opening question
    return JSONResponse({"user": user, "sessions": [sessions[s] for s in order]})
