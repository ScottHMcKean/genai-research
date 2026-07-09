"""Databricks App — chat front-end for the Claims RAG agent.

Serves a custom streaming chat UI that shows the agent's *thinking* (each Vector Search
MCP tool call + result) and the final answer. The agent (agent.py) authenticates as the
app's service principal via WorkspaceClient(); grant it the LLM endpoint + VS index in the
bundle (resources.apps.*.resources).
"""
import json
import pathlib

import mlflow
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from mlflow.types.responses import ResponsesAgentRequest

from agent import AGENT

mlflow.openai.autolog()  # traces every conversation in the app's experiment

app = FastAPI(title="Claims RAG Agent")
_HTML = (pathlib.Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: Request) -> StreamingResponse:
    body = await req.json()
    messages = body.get("input") or [{"role": "user", "content": body.get("message", "")}]

    def event_stream():
        try:
            for ev in AGENT.predict_stream(ResponsesAgentRequest(input=messages)):
                item = ev.item
                if not isinstance(item, dict):
                    item = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:  # surface errors to the UI instead of a dead stream
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
