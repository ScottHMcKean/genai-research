"""Scenario-simulation agent — an MLflow `ResponsesAgent` that answers questions about
simulation runs by calling governed Unity Catalog tools and searching methodology docs.

Tools are reached through two **managed MCP servers**, so there is no tool-wiring glue in
the agent and every call is governed by Unity Catalog:

  - Functions MCP      `/api/2.0/mcp/functions/{catalog}/{schema}`      → the UC SQL tools
      `read_runs`, `read_sim`, `discover_models` (each returns compact JSON).
  - Vector Search MCP  `/api/2.0/mcp/vector-search/{catalog}/{schema}` → methodology-doc RAG.

The LLM decides when to call a tool; we execute it against the right MCP server, feed the
result back, and let the model answer. Intermediate tool-call + tool-output items are yielded
so a chat UI can render the agent's "thinking".

This is a *reference* implementation: it fixes two things the source notebook got wrong.
  1. **No per-request state on the agent instance.** User/session context is read from
     `custom_inputs` and stamped on the trace inside `predict`/`predict_stream` only — never
     cached on `self`, which would leak identity across concurrent app/serving requests.
  2. **Truncation never breaks JSON.** Oversized tool output is wrapped in a valid JSON
     envelope (`cap_tool_output`) instead of being sliced mid-structure.

Kept standalone so `log_model(code_paths=["config.py"])` packages it and the Databricks App
in `app/` can `from agent import AGENT`.
"""
import concurrent.futures
import json
import uuid
from typing import Any, Callable, Generator

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from config import CATALOG, SCHEMA
from config import CHAT_MODEL as LLM_ENDPOINT

# Managed MCP servers for this schema. UC governs every tool call.
MCP_PATHS = [
    f"/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}",
    f"/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}",
]

SYSTEM_PROMPT = (
    "You are a scenario-analysis assistant for a team of modellers. You help them inspect "
    "simulation runs and how metrics respond to scenarios over a sequence of steps.\n"
    "Call tools instead of guessing:\n"
    "- discover_models: list the models available to simulate against.\n"
    "- read_runs: look up simulation runs and their status/metadata (optionally by run_id).\n"
    "- read_sim: read metric responses. Call it FIRST without metrics to discover which "
    "scenarios, metrics, and steps exist for a run, then call it again with exact scenarios "
    "and metrics to get the numbers. Never invent run_ids, scenarios, metrics, or values.\n"
    "- the knowledge-base search tool: retrieve methodology/documentation passages and cite "
    "the source file in square brackets, e.g. [overview.md].\n"
    "Lead with the direct answer, keep numbers rounded and readable, and say plainly when the "
    "tools return nothing rather than fabricating a result."
)
MAX_TURNS = 6           # LLM<->tool round-trips before forcing a final answer
MAX_TOOL_OUTPUT = 6000  # chars; larger tool results are capped — see cap_tool_output


def cap_tool_output(s: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    """Cap an oversized tool result WITHOUT handing the model invalid JSON.

    The source notebook truncated compact JSON at a raw character count, which routinely
    produced half-parsed JSON. Instead, return a small *valid* JSON envelope carrying a
    prefix of the original plus a hint to narrow the query.
    """
    if s is None:
        return ""
    if len(s) <= limit:
        return s
    return json.dumps({
        "truncated": True,
        "original_chars": len(s),
        "hint": "Result too large; showing a prefix. Narrow the query "
                "(fewer runs/scenarios/metrics or a smaller step range).",
        "content_prefix": s[:limit],
    })


def _text(content: Any) -> str:
    """Flatten Responses content (str | list[part] | None) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _to_chat(msg: dict) -> list[dict]:
    """Convert one Responses-API item into Chat-Completions messages for the LLM call."""
    t = msg.get("type")
    if t == "function_call":
        return [{
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": msg["call_id"], "type": "function",
                "function": {"name": msg["name"], "arguments": msg["arguments"]},
            }],
        }]
    if t == "function_call_output":
        return [{"role": "tool", "content": msg["output"], "tool_call_id": msg["call_id"]}]
    if t == "message" and isinstance(msg.get("content"), list):
        return [{"role": msg["role"], "content": _text(msg["content"])}]
    return [{k: v for k, v in msg.items()
             if k in ("role", "content", "name", "tool_calls", "tool_call_id")}]


def _sync(fn: Callable, *args, **kwargs):
    """Run a DatabricksMCPClient call in a worker thread. The client uses asyncio.run
    internally, which raises if a loop is already running (notebooks, the async app
    server); a dedicated thread guarantees a clean loop everywhere."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args, **kwargs).result()


class ScenarioAgent(ResponsesAgent):
    def __init__(self):
        self._ws = None
        self._tool_specs_cache: list[dict] | None = None
        self._tool_client: dict[str, Any] = {}  # tool name -> MCP client (routing map)

    # ---- Databricks / MCP plumbing -------------------------------------------------
    def _workspace(self):
        if self._ws is None:
            from databricks.sdk import WorkspaceClient
            self._ws = WorkspaceClient()
        return self._ws

    def _mcp_client(self, path: str):
        from databricks_mcp import DatabricksMCPClient
        ws = self._workspace()
        url = ws.config.host.rstrip("/") + path
        return DatabricksMCPClient(server_url=url, workspace_client=ws)

    def _tool_specs(self) -> list[dict]:
        """List tools from every MCP server once; cache the specs and a name->client map.

        This map is deterministic configuration (not per-request state), so caching it on
        the instance is safe under concurrency."""
        if self._tool_specs_cache is None:
            specs: list[dict] = []
            for path in MCP_PATHS:
                client = self._mcp_client(path)
                for t in _sync(client.list_tools):
                    specs.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.inputSchema,
                        },
                    })
                    self._tool_client[t.name] = client
            self._tool_specs_cache = specs
        return self._tool_specs_cache

    @mlflow.trace(span_type=SpanType.TOOL)
    def _call_tool(self, name: str, args: dict) -> str:
        client = self._tool_client.get(name)
        if client is None:
            return json.dumps({"error": f"unknown tool: {name}"})
        res = _sync(client.call_tool, name, args)
        out = "".join(getattr(c, "text", "") for c in res.content)
        return cap_tool_output(out)

    def _llm(self, messages: list[dict], tools: list[dict] | None):
        from databricks_openai import DatabricksOpenAI
        kwargs: dict[str, Any] = {"model": LLM_ENDPOINT, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        return DatabricksOpenAI().chat.completions.create(**kwargs)

    # ---- Agent loop ----------------------------------------------------------------
    def _run(self, request: ResponsesAgentRequest) -> Generator[dict, None, None]:
        tools = self._tool_specs()
        history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for inp in request.input:
            history.append(inp.model_dump() if hasattr(inp, "model_dump") else dict(inp))

        for _ in range(MAX_TURNS):
            flat: list[dict] = []
            for m in history:
                flat.extend(_to_chat(m))
            choice = self._llm(flat, tools).choices[0].message
            tool_calls = choice.tool_calls or []

            if not tool_calls:
                yield self.create_text_output_item(text=choice.content or "", id=uuid.uuid4().hex)
                return

            for tc in tool_calls:
                call_id, name, args_json = tc.id, tc.function.name, tc.function.arguments
                yield self.create_function_call_item(
                    id=uuid.uuid4().hex, call_id=call_id, name=name, arguments=args_json)
                history.append({"type": "function_call", "call_id": call_id,
                                "name": name, "arguments": args_json})
                try:
                    output = self._call_tool(name, json.loads(args_json or "{}"))
                except Exception as e:  # surfaced to the model + UI
                    output = json.dumps({"error": f"{type(e).__name__}: {e}"})
                yield self.create_function_call_output_item(call_id=call_id, output=output)
                history.append({"type": "function_call_output", "call_id": call_id, "output": output})

        # Fell through MAX_TURNS: force a final answer without tools.
        flat = []
        for m in history:
            flat.extend(_to_chat(m))
        final = self._llm(flat, None).choices[0].message.content or ""
        yield self.create_text_output_item(text=final, id=uuid.uuid4().hex)

    def _tag_trace(self, request: ResponsesAgentRequest) -> None:
        """Stamp user + session onto the CURRENT trace from custom_inputs.

        Read per request and written straight to the trace — never cached on `self`, so
        concurrent requests cannot read each other's identity. The app forwards the
        authenticated end user here (not the app's service principal)."""
        ci = request.custom_inputs or {}
        meta = {}
        if ci.get("user"):
            meta["mlflow.trace.user"] = str(ci["user"])
        if ci.get("session_id"):
            meta["mlflow.trace.session"] = str(ci["session_id"])
        if meta:
            mlflow.update_current_trace(metadata=meta)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        self._tag_trace(request)
        return ResponsesAgentResponse(output=list(self._run(request)))

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        self._tag_trace(request)
        for item in self._run(request):
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)


try:
    mlflow.openai.autolog()  # auto-trace every LLM call
except Exception:
    pass

AGENT = ScenarioAgent()
mlflow.models.set_model(AGENT)
