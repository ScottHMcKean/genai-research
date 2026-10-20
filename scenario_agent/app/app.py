"""Databricks App — demo front-end for the scenario-simulation agent.

Runs the agent (agent.py) **in-process**: the App's service principal authenticates via
`WorkspaceClient()`; LLM calls go to pay-per-token Foundation Model APIs and tools to the
managed MCP servers. No always-on Model Serving endpoint — cheaper, one less hop.

Four tabs, each a small JSON endpoint:
  - **Chat**   `POST /api/chat`   — streams the agent (SSE), showing each tool step.
  - **Config** `GET  /api/config` — the exact UC/model/MCP wiring the agent uses.
  - **Tools**  `GET  /api/tools` + `POST /api/tool_call` — list the MCP tools and invoke one
               with a **bare MCP call**, returning the raw governed tool output.
  - **Traces** `GET  /api/traces` — recent traces + a link to the MLflow experiment.

Grant the App's service principal: CAN QUERY on the LLM + embedding endpoints, USE
CATALOG/USE SCHEMA + EXECUTE on the UC functions (and SELECT on the tables they read), and
SELECT on the Vector Search index. The authenticated END USER is forwarded into the trace
(`X-Forwarded-Email`) so conversations are attributable to a person, not the SP.
"""
import json
import pathlib

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from mlflow.tracking import MlflowClient
from mlflow.types.responses import ResponsesAgentRequest

import config
from agent import AGENT, MCP_PATHS, _sync

_ws = WorkspaceClient()
HOST = _ws.config.host.rstrip("/")

mlflow.openai.autolog()  # trace every conversation

# Point traces at a stable, linkable experiment (best-effort — the app still runs if this
# can't be set; the Traces tab just won't have a deep link).
EXPERIMENT_ID = None
EXPERIMENT_URL = None
try:
    _me = _ws.current_user.me().user_name
    _exp = mlflow.set_experiment(f"/Users/{_me}/scenario_agent_app")
    EXPERIMENT_ID = _exp.experiment_id
    EXPERIMENT_URL = f"{HOST}/ml/experiments/{EXPERIMENT_ID}"
except Exception:
    pass

app = FastAPI(title="Scenario Assistant")
_HTML = (pathlib.Path(__file__).parent / "static" / "index.html").read_text()


def _mcp(path: str):
    from databricks_mcp import DatabricksMCPClient
    return DatabricksMCPClient(server_url=HOST + path, workspace_client=_ws)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Config tab ──────────────────────────────────────────────────────────────
@app.get("/api/config")
def api_config() -> dict:
    return {
        "catalog": config.CATALOG,
        "schema": config.SCHEMA,
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
        "vector_search_endpoint": config.VS_ENDPOINT,
        "vector_search_index": config.VS_INDEX,
        "uc_function_tools": config.UC_TOOL_FUNCTIONS,
        "mcp_servers": [HOST + p for p in MCP_PATHS],
        "workspace_host": HOST,
        "mlflow_experiment": EXPERIMENT_URL,
    }


# ── Tools tab (bare MCP) ──────────────────────────────────────────────────────
@app.get("/api/tools")
def api_tools() -> dict:
    servers = []
    for path in MCP_PATHS:
        kind = "functions" if "/functions/" in path else "vector-search"
        entry = {"server": kind, "url": HOST + path, "tools": []}
        try:
            for t in _sync(_mcp(path).list_tools):
                entry["tools"].append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                })
        except Exception as e:  # surface MCP/permission errors to the UI
            entry["error"] = f"{type(e).__name__}: {e}"
        servers.append(entry)
    return {"servers": servers}


@app.post("/api/tool_call")
async def api_tool_call(req: Request) -> JSONResponse:
    """Invoke one MCP tool directly (a bare MCP call) and return its raw output."""
    body = await req.json()
    name = body.get("name")
    args = body.get("args") or {}
    last_err = None
    for path in MCP_PATHS:
        try:
            client = _mcp(path)
            if name in {t.name for t in _sync(client.list_tools)}:
                res = _sync(client.call_tool, name, args)
                raw = "".join(getattr(c, "text", "") for c in res.content)
                return JSONResponse({"name": name, "server": HOST + path, "raw": raw})
        except Exception as e:  # don't give up — the tool may live on another server
            last_err = f"{type(e).__name__}: {e}"
            continue
    if last_err:
        return JSONResponse({"error": last_err}, status_code=500)
    return JSONResponse({"error": f"tool '{name}' not found on any MCP server"}, status_code=404)


# ── Traces tab ────────────────────────────────────────────────────────────────
@app.get("/api/traces")
def api_traces() -> dict:
    if not EXPERIMENT_ID:
        return {"experiment_url": None, "traces": [],
                "note": "No experiment bound; traces still log to the app's default experiment."}
    out = []
    try:
        for t in MlflowClient().search_traces(experiment_ids=[EXPERIMENT_ID], max_results=25):
            info = t.info
            tid = getattr(info, "trace_id", None) or getattr(info, "request_id", "")
            out.append({
                "trace_id": tid,
                "timestamp_ms": getattr(info, "timestamp_ms", None) or getattr(info, "request_time", None),
                "status": str(getattr(info, "status", getattr(info, "state", ""))),
                "execution_ms": getattr(info, "execution_time_ms", None) or getattr(info, "execution_duration", None),
                "url": f"{EXPERIMENT_URL}/traces?selectedTraceId={tid}" if tid else EXPERIMENT_URL,
            })
    except Exception as e:
        return {"experiment_url": EXPERIMENT_URL, "error": f"{type(e).__name__}: {e}", "traces": []}
    return {"experiment_url": EXPERIMENT_URL, "traces": out}


# ── Chat tab ──────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: Request) -> StreamingResponse:
    body = await req.json()
    messages = body.get("input") or [{"role": "user", "content": body.get("message", "")}]
    user = (req.headers.get("X-Forwarded-Email")
            or req.headers.get("X-Forwarded-User") or "unknown")
    session_id = body.get("session_id") or "web"

    def event_stream():
        # Open an explicit root span for the turn. Served models get an auto root span, but
        # here the agent runs in-process, so without this the user/session tags (and the
        # autologged LLM calls) would have no single trace to attach to.
        with mlflow.start_span(name="chat_turn", span_type=SpanType.AGENT):
            mlflow.update_current_trace(
                metadata={"mlflow.trace.user": user, "mlflow.trace.session": session_id})
            try:
                for ev in AGENT.predict_stream(ResponsesAgentRequest(
                    input=messages, custom_inputs={"user": user, "session_id": session_id},
                )):
                    item = ev.item
                    if not isinstance(item, dict):
                        item = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
                    yield f"data: {json.dumps(item)}\n\n"
            except Exception as e:  # surface errors to the UI instead of a dead stream
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
