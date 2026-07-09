# Claims RAG agent — on Databricks Apps

The same RAG agent as `agents/agent.py`, shipped as a **Databricks App** with a custom
streaming chat UI that shows the agent's *thinking* — each Vector Search (MCP) tool call and
its result — and the final, cited answer.

## Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI server — serves the chat UI at `/` and streams the agent at `POST /api/chat` (SSE) |
| `agent.py` | The `ResponsesAgent` (Vector Search via the managed MCP server); a copy of `agents/agent.py` so the app is self-contained |
| `config.py` | Catalog/schema/model (edit to point at your own data) |
| `static/index.html` | Custom vanilla-JS chat UI (no build step) that renders tool-call steps + answer |
| `app.yaml` | App runtime command (`uvicorn app:app --port 8000`) |
| `requirements.txt` | `databricks-mcp`, `databricks-openai`, `databricks-sdk[openai]`, `mcp`, `mlflow` |

## Run it

Deploy via the bundle from the repo root (the app runs as its own service principal; the
bundle grants it the LLM endpoint + the Vector Search index):

```bash
databricks bundle deploy -t dev
databricks bundle run   -t dev claims_rag_app
```

The command prints the app URL (`https://<app>.databricksapps.com`). Prerequisite: run
`fins_data/generate_data.py` and `agents/01_vector_search.py` first so the index exists.

## How it works

- The UI POSTs the conversation to `/api/chat`; the server runs `AGENT.predict_stream(...)`
  and forwards each Responses-API event as an SSE line.
- `function_call` / `function_call_output` events render as collapsible **search steps**;
  the final `message` renders as the answer (with source citations).
- Auth: `WorkspaceClient()` picks up the app's service-principal credentials; the managed
  Vector Search MCP server enforces Unity Catalog permissions on the index.
