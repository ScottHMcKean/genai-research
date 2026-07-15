# Agent Bricks + Custom RAG — Insurance Claims (Section 2 "Agents")

A simple, from-scratch demo of **building agents on Databricks** over an insurance
claims spine. Two complementary approaches, side by side:

1. **Agent Bricks** — governed, managed tiles (Knowledge Assistant + Supervisor),
   created entirely in code via the Databricks SDK.
2. **A custom RAG agent** — an MLflow `ResponsesAgent` over Vector Search, deployed to
   Model Serving. Full control when you need it.

All on synthetic P&C claims data (no real customer data). Insurance & claims themed
throughout.

## Talk track

- **"You don't have to choose between managed and custom."** Agent Bricks gets a
  governed Q&A + multi-agent supervisor stood up in minutes; the custom RAG agent shows
  the same retrieval built explicitly when you need bespoke logic, and both deploy as
  governed serving endpoints under Unity Catalog.
- **Retrieval is grounded and cited** — the agent answers strictly from policy wordings
  and claims-handling guidelines, and cites the source document.
- **The Supervisor routes** policy questions to the Knowledge Assistant and
  claim-lookup questions to a Unity Catalog function — governed tools, not glue code.

## Notebooks (run in order)

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_setup.py` | Checks the shared claims spine exists (run `fins_data/generate_data.py` first). |
| 01 | `01_vector_search.py` | Creates the `claims_vs` endpoint if needed and builds a **Delta-Sync Vector Search index** on `doc_chunks` with managed embeddings (`databricks-gte-large-en`). |
| 02 | `02_custom_rag_agent.py` | Logs `agent.py` (a `ResponsesAgent`), registers to Unity Catalog, **deploys a Model Serving endpoint**, and **populates 10 traces**. |
| 03 | `03_agent_bricks.py` | Creates an **Agent Bricks Knowledge Assistant**, a **UC function** claim-lookup tool, and a **Supervisor Agent** that routes between them (+ 10 traces). |
| 04 | `04_agent_bricks_monitoring.ipynb` | Monitors a deployed Agent Bricks / serving endpoint via **system tables** — `system.serving.served_entities`, `endpoint_usage`, and `system.billing.usage` for per-endpoint DBU cost. (Endpoint usage for Agent Bricks is roadmap; billing works today.) |

`agent.py` — the custom RAG agent. It retrieves by calling the **managed Vector Search MCP
server** (`/api/2.0/mcp/vector-search/{catalog}/{schema}`) as a tool, so the LLM decides when
to search and the agent returns the intermediate tool-call steps (its "thinking") with the
cited answer. Kept standalone so `mlflow.pyfunc.log_model(code_paths=["config.py"])` packages it.

## Ship it as an app — [`app/`](app/README.md)

`app/` deploys the **same agent as a Databricks App** (agents on Apps) via the bundle
(`resources/agents_app.yml` → `claims_rag_app`), with a **custom streaming chat UI** that
shows each Vector Search step and the answer. `databricks bundle run -t dev claims_rag_app`.

## Platform capabilities showcased

Vector Search via **managed MCP tool calling** · MLflow 3 `ResponsesAgent` + auto tracing ·
Unity Catalog model registry · one-line `agents.deploy` · **agents on Databricks Apps** with
a custom chat UI · Agent Bricks (KA + Supervisor) · UC functions as governed tools.

## Run it

1. Run `fins_data/generate_data.py` once (builds the common data).
2. Open this folder and run `00 → 03` in order on serverless.

Optional — deploy from the repo root as a Job instead:

```bash
databricks bundle deploy -t dev
databricks bundle run -t dev agents_job
```

## Notes

- **Config** is in `config.py` (catalog/schema, models, VS endpoint). Defaults to
  `shm_skunkworks_catalog.claims_demo` (edit `config.py` for your own catalog/schema).
- **Cost**: pay-per-token FMAPI + a small serverless VS index + one serving endpoint.
  Delete the `claims-rag-agent` endpoint and the KA/Supervisor tiles when done.
- **Genie**: natural-language SQL over the claims table is shown in the **AI Governance**
  demo via a managed Genie MCP server. (The Genie `create_space` SDK path has a
  `serialized_space` serialization bug in the current build, so this demo routes the
  Supervisor's structured-data tool through a UC function instead.)
- The KA and Supervisor take a few minutes to provision to `ACTIVE`; the custom RAG
  serving endpoint takes ~15 minutes to reach `READY`.
