"""Claims RAG agent — MLflow ResponsesAgent over Databricks Vector Search via the
managed **MCP** server (`/api/2.0/mcp/vector-search/{catalog}/{schema}`).

The LLM decides when to call the Vector Search tool; we execute it against the managed
MCP server, feed the results back, and let the model answer. The agent returns the
intermediate tool-call + tool-output items alongside the final answer, so a chat UI can
render the "thinking" / retrieval steps. Streaming is supported via `predict_stream`.

Single source of truth for config is `config.py` (packaged via log_model code_paths).
"""
import concurrent.futures
import json
import uuid
from typing import Any, Callable, Generator

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from config import CHAT_MODEL as LLM_ENDPOINT, CATALOG, SCHEMA

# Managed Vector Search MCP server for this schema. It exposes one tool per index:
# `{catalog}__{schema}__{index}` taking {"query": <text>} and returning the top chunks.
VS_MCP_PATH = f"/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"

SYSTEM_PROMPT = (
    "You are a claims assistant for a property & casualty insurer. Use the Vector Search "
    "tool to retrieve the relevant policy wordings and claims-handling guidelines, then "
    "answer strictly from what you retrieve and cite the source document in square "
    "brackets, e.g. [property_water_damage_procedure.md]. If the retrieved context does "
    "not contain the answer, say you don't have that information."
)
MAX_TURNS = 4  # LLM<->tool round-trips before forcing a final answer


def _sync(fn: Callable, *args, **kwargs):
    """Run a DatabricksMCPClient call in a worker thread. The client uses asyncio.run
    internally, which raises if a loop is already running (notebooks, the async app
    server); a dedicated thread guarantees a clean loop everywhere."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args, **kwargs).result()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _to_chat(msg: dict) -> list[dict]:
    """Convert a Responses-API item into Chat-Completions messages for the LLM call."""
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
        return [{"role": msg["role"], "content": "".join(c.get("text", "") for c in msg["content"])}]
    return [{k: v for k, v in msg.items() if k in ("role", "content", "name", "tool_calls", "tool_call_id")}]


class ClaimsRAGAgent(ResponsesAgent):
    def __init__(self):
        self._ws = None
        self._tools_cache = None

    # ---- Databricks / MCP plumbing -------------------------------------------------
    def _workspace(self):
        if self._ws is None:
            from databricks.sdk import WorkspaceClient
            self._ws = WorkspaceClient()
        return self._ws

    def _mcp_client(self):
        from databricks_mcp import DatabricksMCPClient
        ws = self._workspace()
        url = ws.config.host.rstrip("/") + VS_MCP_PATH
        return DatabricksMCPClient(server_url=url, workspace_client=ws)

    def _tool_specs(self) -> list[dict]:
        if self._tools_cache is None:
            tools = _sync(self._mcp_client().list_tools)
            self._tools_cache = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "Search the claims knowledge base.",
                    "parameters": t.inputSchema,
                },
            } for t in tools]
        return self._tools_cache

    @mlflow.trace(span_type="RETRIEVER")
    def _call_tool(self, name: str, args: dict) -> str:
        res = _sync(self._mcp_client().call_tool, name, args)
        return "".join(getattr(c, "text", "") for c in res.content)

    def _llm(self, messages: list[dict], tools: list[dict] | None):
        from databricks_openai import DatabricksOpenAI
        kwargs = {"model": LLM_ENDPOINT, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        return DatabricksOpenAI().chat.completions.create(**kwargs)

    # ---- Agent loop ----------------------------------------------------------------
    def _run(self, request: ResponsesAgentRequest):
        """Run the tool-calling loop, yielding Responses-API output items as they occur."""
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
                item = self.create_text_output_item(text=choice.content or "", id=uuid.uuid4().hex)
                history.append({"type": "message", "role": "assistant",
                                "content": [{"type": "output_text", "text": choice.content or ""}]})
                yield item
                return

            for tc in tool_calls:
                name, args_json = tc.function.name, tc.function.arguments
                call_id = tc.id
                yield self.create_function_call_item(
                    id=uuid.uuid4().hex, call_id=call_id, name=name, arguments=args_json)
                history.append({"type": "function_call", "call_id": call_id,
                                "name": name, "arguments": args_json})
                try:
                    output = self._call_tool(name, json.loads(args_json or "{}"))
                except Exception as e:  # tool failure is surfaced to the model + UI
                    output = f"Error calling {name}: {e}"
                yield self.create_function_call_output_item(call_id=call_id, output=output)
                history.append({"type": "function_call_output", "call_id": call_id, "output": output})

        # Fell through MAX_TURNS: force a final answer without tools.
        flat = []
        for m in history:
            flat.extend(_to_chat(m))
        final = self._llm(flat, None).choices[0].message.content or ""
        yield self.create_text_output_item(text=final, id=uuid.uuid4().hex)

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        return ResponsesAgentResponse(output=list(self._run(request)))

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # Emit each item (tool calls, tool outputs, final message) as it's produced so
        # the chat UI can render intermediate retrieval steps live.
        for item in self._run(request):
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)


mlflow.openai.autolog()
AGENT = ClaimsRAGAgent()
mlflow.models.set_model(AGENT)
