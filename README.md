# GenAI Notebooks

Index of notebooks by topic. Most run on [Databricks Serverless](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/four) (set **Base environment** to env 4 in the notebook **Environment** panel).

---

## FINS demo suite

A cohesive demo suite on **one common Unity Catalog dataset** (financial-services /
insurance claims — chosen only to illustrate the features). Each use case is its own
folder of **notebooks you run in order** on serverless; a customer can lift a single
folder, point `config.py` at their own data, and go.

**Run order:** run `fins_data/generate_data.py` **once** to build the common dataset, then
open any use-case folder and run its notebooks `00 → 03`.

| Folder | Use case | Showcases |
|--------|----------|-----------|
| [`fins_data/`](fins_data/README.md) | **Common data** (run first) | One script → synthetic claims (+PII), adjuster notes, knowledge docs, chunked VS source, real public insurance PDFs |
| [`agents/`](agents/README.md) | **Agents** | Agent Bricks (Knowledge Assistant + Supervisor), custom RAG agent (Vector Search via **MCP tool calling**), model serving, and the **same agent shipped as a Databricks App** ([`agents/app/`](agents/app/README.md)) with a streaming chat UI |
| [`governance/`](governance/README.md) | **Governance** | Unity AI Gateway (usage, rate limits, **PII guardrails**, inference tables), managed + external MCP, lineage & cost observability |
| [`ai_runtime/`](ai_runtime/README.md) | **AI Runtime** | Ray on serverless (fan-out + Ray Data), LoRA fine-tune on **serverless GPU** → UC model registry, fine-tuned vs zero-shot eval |
| [`document_intelligence/`](document_intelligence/README.md) | **Document Intelligence** | `ai_parse_document` → `ai_extract` → MLflow evaluate over the common insurance PDFs |

Everything runs as plain notebooks. Optional: `databricks bundle deploy -t dev` deploys
each folder as a Job (see `resources/*.yml`) if you'd rather run them from the root.

---

## Index

### AI functions (benchmarking & testing, incl. AI_QUERY and external models)
| Notebook |
|----------|
| [ai_functions/les_mis_data_prep.ipynb](ai_functions/les_mis_data_prep.ipynb) |
| [ai_functions/les_mis_endpoint_creation.ipynb](ai_functions/les_mis_endpoint_creation.ipynb) |
| [ai_functions/les_miserables_aiquery.ipynb](ai_functions/les_miserables_aiquery.ipynb) |
| [ai_functions/les_miserables_ray.ipynb](ai_functions/les_miserables_ray.ipynb) |
| [ai_functions/les_miserables_spark.ipynb](ai_functions/les_miserables_spark.ipynb) |
| [ai_functions/ray_external_model_les_mis.ipynb](ai_functions/ray_external_model_les_mis.ipynb) |

### External models
| Notebook |
|----------|
| [external_models/azure_assistant_tracing.ipynb](external_models/azure_assistant_tracing.ipynb) |
| [external_models/azure_search_responses_agent.ipynb](external_models/azure_search_responses_agent.ipynb) |
| [external_models/openai_oss_thinking.ipynb](external_models/openai_oss_thinking.ipynb) |

### Cost management
| Notebook |
|----------|
| [cost_management/throughput_estimation.ipynb](cost_management/throughput_estimation.ipynb) |

### Custom models
| Notebook |
|----------|
| [custom_models/Gemma3.ipynb](custom_models/Gemma3.ipynb) |
| [custom_models/Qwen Example.ipynb](custom_models/Qwen%20Example.ipynb) |
| [custom_models/tinyllama_transformers.ipynb](custom_models/tinyllama_transformers.ipynb) |

### DSPy
| Notebook |
|----------|
| [dspy/azure_search_extraction.ipynb](dspy/azure_search_extraction.ipynb) |

### Evaluation & MLflow
| Notebook |
|----------|
| [mlflow/arxiv_eval_walkthrough.ipynb](mlflow/arxiv_eval_walkthrough.ipynb) |
| [mlflow/rest_api_walkthrough.ipynb](mlflow/rest_api_walkthrough.ipynb) |
| [evaluation/mlflow_genai_evaluation.ipynb](evaluation/mlflow_genai_evaluation.ipynb) |

End-to-end MLflow 3 GenAI evaluation walkthrough (`mlflow/`): a LangGraph
ReAct agent over the public arXiv API that exercises all eight stages of the
Databricks MLflow UI flow — Trace, Sessions, Judges, Evaluation Datasets, Evaluation
Runs, Labeling Schemas, Labeling Sessions, and Prompts / Agent Versioning. See
`mlflow/README.md` for the opinionated writeup. The companion
`rest_api_walkthrough.ipynb` is a REST-only version of the same flow for
non-Python frameworks (C#, Java, Go) — every call is `requests`-based, ports
directly to `HttpClient`/`OkHttp`/`net/http`, and OTLP traces it emits show up
in the same MLflow Traces UI as SDK-emitted ones.

### FastAPI
| Notebook |
|----------|
| [fastapi/test_app.ipynb](fastapi/test_app.ipynb) |
| [fastapi/test_llm_client.ipynb](fastapi/test_llm_client.ipynb) |

### LangGraph
| Notebook |
|----------|
| [langgraph/langgraph_basics.ipynb](langgraph/langgraph_basics.ipynb) |
| [langgraph/reasoning.ipynb](langgraph/reasoning.ipynb) |
| [langgraph/structured_outputs.ipynb](langgraph/structured_outputs.ipynb) |

### MCP (Model Context Protocol)
| Notebook |
|----------|
| [mcp/test_connection.ipynb](mcp/test_connection.ipynb) |

### Document Intelligence
| Notebook |
|----------|
| [document_intelligence/claim_doc_ai_parse.ipynb](document_intelligence/claim_doc_ai_parse.ipynb) |
| [document_intelligence/claim_doc_ray_claude.ipynb](document_intelligence/claim_doc_ray_claude.ipynb) |
| [document_intelligence/claim_doc_profile.ipynb](document_intelligence/claim_doc_profile.ipynb) |
| [document_intelligence/ai_query.dbquery.ipynb](document_intelligence/ai_query.dbquery.ipynb) |
| [document_intelligence/claude_ai_query.ipynb](document_intelligence/claude_ai_query.ipynb) |
| [document_intelligence/claude_structured.ipynb](document_intelligence/claude_structured.ipynb) |
| [document_intelligence/docling_on_databricks.ipynb](document_intelligence/docling_on_databricks.ipynb) |
| [document_intelligence/few_shot_multimodal_classification.ipynb](document_intelligence/few_shot_multimodal_classification.ipynb) |
| [document_intelligence/pdf_parsing.ipynb](document_intelligence/pdf_parsing.ipynb) |

Two complementary PDF extraction paths on a hand-labeled golden set of 10 public
insurance documents, scored with the same tolerant scorer and shared JSON schema:

- `claim_doc_ai_parse.ipynb` — SQL-native `ai_parse_document` → `ai_query` chain,
  Delta-backed prompt registry.
- `claim_doc_ray_claude.ipynb` — direct FMAPI `DocumentContent` call (the path
  needed because `ai_query` still doesn't accept PDF bytes as of 2026-04),
  parallelized with Ray on Serverless env v5.

The remaining notebooks cover adjacent multimodal/document-parsing territory (image
→ text via Claude, Docling serving, few-shot classification, PDF-to-Claude via the
Anthropic SDK).

### vLLM
| Notebook |
|----------|
| [vllm/qwen35_4b_throughput.ipynb](vllm/qwen35_4b_throughput.ipynb) |

vLLM + Qwen 3.5 4B throughput profiling (~500 rows). Deploy and run via Asset Bundle on single-node A100 (`NC24ads_A100_v4`), ML Runtime 16.4 LTS: `databricks bundle deploy -t dev` then `databricks bundle run -t dev vllm_qwen_throughput_job`.

