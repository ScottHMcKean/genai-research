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

Deploy as a Databricks App with the CLI (`databricks apps`). Create the app once, sync this
folder, then deploy:

```bash
databricks apps create claims-rag-agent          # once
databricks sync . /Workspace/Users/<you>/claims-rag-agent
databricks apps deploy claims-rag-agent --source-code-path /Workspace/Users/<you>/claims-rag-agent
```

`databricks apps list` prints the app URL (`https://<app>.databricksapps.com`). The app runs
as its own service principal — grant that principal the LLM serving endpoint (`CAN QUERY`) and
the Vector Search index (`USE` on catalog/schema + the index). Prerequisite: run
`fins_data/generate_data.py` and `agents/01_vector_search.py` first so the index exists.

## How it works

- The UI POSTs the conversation to `/api/chat`; the server runs `AGENT.predict_stream(...)`
  and forwards each Responses-API event as an SSE line.
- `function_call` / `function_call_output` events render as collapsible **search steps**;
  the final `message` renders as the answer (with source citations).
- Auth: `WorkspaceClient()` picks up the app's service-principal credentials; the managed
  Vector Search MCP server enforces Unity Catalog permissions on the index.
