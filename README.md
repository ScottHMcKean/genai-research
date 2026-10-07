# GenAI Research

A collection of **self-contained Databricks GenAI demos and benchmarks**. Each folder is
independent: open it and run its notebooks **in order** (`00 → 0N`) on
[Databricks Serverless](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/four)
(set **Base environment** to the latest env in the notebook **Environment** panel). Most
folders have their own `README.md` and a local `config.py` you can point at your own data.

> **No asset bundle.** This repo used to ship a root Databricks Asset Bundle (`databricks.yml`
> + `resources/*.yml`) that deployed every folder as a Job. It was removed — the demos are
> independent and are meant to be run as notebooks. To run one headless, use the notebook
> Jobs UI or `databricks jobs submit` against the folder's notebooks.

## How to run

1. Install + authenticate the Databricks CLI: `databricks auth login --host <workspace> --profile <name>`.
2. Clone this repo into your workspace (Repos / Git folder), or sync a folder with
   `databricks workspace import-dir <folder> /Workspace/Users/<you>/<folder>`.
3. Open a folder, read its `README.md`, and run the notebooks in numeric order on serverless.

---

## Demo suites

**FINS suite** — a cohesive set of use cases on **one common Unity Catalog dataset**
(financial-services / insurance claims). Run [`fins_data/`](fins_data/README.md) **once** to
build the shared dataset, then run any use-case folder `00 → 03`.

| Folder | Use case | Showcases |
|--------|----------|-----------|
| [`fins_data/`](fins_data/README.md) | **Common data** (run first) | One script → synthetic claims (+PII), adjuster notes, knowledge docs, chunked VS source, real insurance PDFs |
| [`agents/`](agents/README.md) | **Agents** | Agent Bricks (Knowledge Assistant + Supervisor), custom RAG agent (Vector Search via **MCP tool calling**), model serving, shipped as a **Databricks App** ([`agents/app/`](agents/app/README.md)) |
| [`governance/`](governance/README.md) | **Governance** | Unity AI Gateway (usage, rate limits, **PII guardrails**, inference tables), managed + external MCP, lineage & cost observability |
| [`ai_runtime/`](ai_runtime/README.md) | **AI Runtime** | Ray on serverless (fan-out + Ray Data), LoRA fine-tune on **serverless GPU** → UC model registry, fine-tuned vs zero-shot eval |
| [`document_intelligence/`](document_intelligence/README.md) | **Document Intelligence** | `ai_parse_document` → `ai_extract` → MLflow evaluate over insurance PDFs |

**Ray walk-through** — a single narrative from Ray basics → distributed inference →
reinforcement learning. Full detail in [`ray/README.md`](ray/README.md).

| # | Notebook / folder | What it shows | Compute |
|---|-------------------|---------------|---------|
| 01 | `ray/01_ray_basics_classic_cluster.ipynb` | Spin up a Ray cluster on a classic Spark cluster, fan work out | Classic cluster |
| 02 | `ray/02_ray_external_model_inference.ipynb` | Task-based batch inference against an OpenAI-compatible model | Classic or serverless |
| 03 | [`ray/03_rl_slm_orchestration/`](ray/03_rl_slm_orchestration/README.md) | GRPO-train a Qwen3 SLM orchestrator with NeMo Gym on AI Runtime + Ray | Serverless GPU |

**Tracing & observability** — [`tracing/`](tracing/README.md) makes tracing real for a custom
agent end to end: framework autolog, OpenTelemetry traces stored in **Unity Catalog**,
conversation history from traces, and per-user **governance**, shipped as a Databricks App.
See [`tracing/WIRING.md`](tracing/WIRING.md) for the components/queries/latencies reference.

---

## Observability & gateway

| Folder | What's in it |
|--------|--------------|
| [`litellm/`](litellm/README.md) | litellm ↔ Databricks FMAPI, litellm **Zerobus** trace logging → UC Delta, an **AI Gateway ⇄ litellm** trace adapter (SQL view + Python + OTel check), and an MLflow trace backfill |
| [`mlflow/`](mlflow/README.md) | MLflow 3 GenAI eval walkthrough on a LangGraph arXiv ReAct agent (all 8 UI stages); `rest_api_walkthrough.ipynb` is the REST-only port for non-Python frameworks |
| [`ai_gateway/`](ai_gateway/) | Calling models through the Mosaic AI Gateway (unified endpoint access, load generation) |
| [`guardrails/`](guardrails/) | Guardrail evaluation two ways — online (Gateway guardrails) and offline (LLM-as-judge + MLflow eval); precision/recall/FPR by attack technique |

## Serving & performance

| Folder | What's in it |
|--------|--------------|
| [`pii_serving/`](pii_serving/README.md) | PII scrubbing via vLLM entrypoint serving — Qwen3-4B / LoRA / Gemma-27B sweep on serverless GPU |
| [`embedding_serving/`](embedding_serving/README.md) | Bioclinical ModernBERT embedding serving with TEI + throughput profiling ([`REPORT.md`](embedding_serving/REPORT.md)) |
| [`vllm/`](vllm/) | vLLM PII-detection throughput profiling |
| [`async_load_test/`](async_load_test/) | Async fire-rate profiler — serverless CPU concurrency sizing |

## Techniques & building blocks

| Folder | What's in it |
|--------|--------------|
| [`vector_search/`](vector_search/) | VS benchmarking: self-managed vs managed-embedding indexes, metadata filters, Databricks VS vs pgvector vs Snowflake Cortex, Spark-side vector ops |
| [`dspy/`](dspy/) | DSPy on Databricks — intro, custom modules/tool-calling, GEPA optimizer, RAG |
| [`ai_functions/`](ai_functions/) | AI functions (`ai_query`) prompt patterns + Les-Misérables benchmarking examples |
| [`external_models/`](external_models/) | Azure OpenAI integration — assistant tracing, responses agent, pyfunc retrievers |
| [`mcp/`](mcp/) | Model Context Protocol — connection test + OpenAI MCP tool-calling agent |
| [`fastapi/`](fastapi/) | Minimal FastAPI app + LLM client test harness |

---

Dependencies are pinned in `pyproject.toml` / `requirements.txt` (`uv.lock` for a reproducible
env). Each demo installs anything extra it needs with `%pip install` at the top of its first
notebook.
