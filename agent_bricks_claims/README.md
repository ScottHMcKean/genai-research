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
| 00 | `00_setup.py` | Checks the shared claims spine exists (run `claims_demo/00_setup_and_data.py` first). |
| 01 | `01_vector_search.py` | Builds a **Delta-Sync Vector Search index** on `doc_chunks` with managed embeddings (`databricks-gte-large-en`), reusing the `vs` endpoint. |
| 02 | `02_custom_rag_agent.py` | Logs `agent.py` (a `ResponsesAgent`) with VS + FMAPI resources, registers to Unity Catalog, and **deploys a Model Serving endpoint** (`claims-rag-agent`). |
| 03 | `03_genie_ka_supervisor.py` | Creates an **Agent Bricks Knowledge Assistant** over the docs, a **UC function** claim-lookup tool, and a **Supervisor Agent** that routes between them. |

`agent.py` — the custom RAG agent (retrieve-then-generate: Vector Search → Claude Sonnet
4.5, grounded + cited). Kept as a standalone file so `mlflow.pyfunc.log_model` can package it.

## Platform capabilities showcased

Vector Search (serverless, managed embeddings, Delta Sync) · MLflow 3 `ResponsesAgent` ·
Unity Catalog model registry · one-line agent deployment (`agents.deploy`) · Agent Bricks
Knowledge Assistant + Supervisor · Unity Catalog functions as governed agent tools ·
foundation-model APIs (Claude Sonnet 4.5).

## Run it

```bash
# from repo root
databricks bundle deploy -t dev
databricks bundle run -t dev claims_demo_setup_job       # build the shared data first
databricks bundle run -t dev agent_bricks_claims_job     # this demo
```

Or run the notebooks interactively in order.

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
