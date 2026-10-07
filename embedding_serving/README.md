# Bioclinical ModernBERT embedding serving + throughput profiling

Deploy and profile the fine-tuned bioclinical embedding model
[`NeuML/bioclinical-modernbert-base-embeddings`](https://huggingface.co/NeuML/bioclinical-modernbert-base-embeddings)
(ModernBERT-base, 768-dim) on Databricks, and compare serving/runtime paths for
**embedding throughput (tokens/s)** on **A10** (primary) vs **T4** (comparison).

> **📊 Results, findings, and the full write-up are in [`REPORT.md`](REPORT.md)** — including the
> throughput matrix (vLLM on A10 ≈ **89.5k tok/s**, the fastest path), the vLLM build/serving story, the
> TEI blocker, and notes on the `llm/v1/embeddings` entrypoint availability.

## Why this exists

Produces apples-to-apples tokens/s numbers for a fine-tuned ModernBERT embedding model across serving
paths on Databricks (AI Runtime baseline, vLLM, sentence-transformers, TEI) on A10 vs T4, to pick the
best way to serve it.

## Key constraints (from research)

- **ModernBERT wants Flash-Attention 2 (Ampere SM 8.0+).** A10 (SM 8.6) gets full FA2; T4
  (Turing SM 7.5) falls back to SDPA / TEI's experimental `turing` image (FA2 off) — expect
  ~40–50% lower throughput and minor precision caveats on T4.
- **Custom LLM Serving (the vLLM express-entrypoint path):** docs indicate `task:
  "llm/v1/embeddings"` with vLLM `--runner pooling` (0.24+) / `--task embed` is now supported.
  We test it directly; if endpoint routing rejects it we fall back to a pyfunc-wrapped in-process
  vLLM engine on standard GPU serving.
- **TEI is not available as a Model Serving container here** (only vLLM custom serving is enabled),
  so TEI is benchmarked by running the TEI server on **AI Runtime GPU compute**, not as an endpoint.
- **Serverless GPU offers A10/H100, not T4.** A10 runs on serverless GPU; T4 baselines run on a
  classic single-node `g4dn` GPU cluster; the vLLM T4-vs-A10 comparison uses Model Serving
  `workload_type` (`GPU_SMALL`=T4, `GPU_MEDIUM`=A10).

## Notebooks

| Notebook | What it does | Compute |
|---|---|---|
| `00_setup_and_data.py` | Create `genai` schema + `huggingface` volume, snapshot the model, download `qiaojin/PubMedQA`, land a clean benchmark corpus table + token stats. | Serverless CPU |
| `01_airuntime_baseline.py` | Load the model with sentence-transformers on GPU, sweep batch size, report **encode tokens/s** (raw runtime ceiling, no serving overhead). | Serverless GPU (A10) |
| `02_vllm_serving.py` | **Working vLLM path** — pyfunc-wrapped in-process `vllm.LLM(runner="pooling")`, registered **without** `env_pack` (server-side build), deploy + benchmark. A10 & T4. | Serverless GPU (build) + Model Serving |
| `02b_st_pyfunc_serving.py` | sentence-transformers pyfunc endpoint (simple, portable). A10 & T4. | Serverless GPU (build) + Model Serving |
| `02c_vllm_entrypoint.py` | vLLM via the Custom LLM Serving **entrypoint** — documents the `llm/v1/embeddings` ring gate; ready once enrolled. | Serverless GPU (build) + Model Serving |
| `03_tei_air.py` | Attempt the HF TEI server on serverless GPU — documents the docker/toolchain blocker. | Serverless GPU (A10) |
| `04_bench_endpoint.py` | Standalone endpoint benchmark (decoupled from deploy; client-side truncation for vLLM). | Serverless CPU |

All notebooks write results to the MLflow experiment and a Delta results table so the three paths
can be compared in one place (`shm_skunkworks_catalog.genai.embedding_bench_results`).

## Config (shared)

- Catalog / schema / volume: `shm_skunkworks_catalog` / `genai` / `huggingface`
- Model: `NeuML/bioclinical-modernbert-base-embeddings` → snapshot at
  `/Volumes/shm_skunkworks_catalog/genai/huggingface/bioclinical_modernbert`
- Corpus table: `shm_skunkworks_catalog.genai.pubmedqa_corpus`
- Results table: `shm_skunkworks_catalog.genai.embedding_bench_results`

## Prior art

- Serving templates adapted from the HuggingFace embedding/chat serving-on-AI-Runtime
  patterns (`hf_embedding_serving_air.py` / `hf_chat_serving_air.py`).
