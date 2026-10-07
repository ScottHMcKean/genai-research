# PII scrubbing via vLLM entrypoint serving

Detect and redact PII in financial documents using open-weight LLMs served on **Databricks Model
Serving with the Custom LLM Serving (vLLM) entrypoint route**, then move data across catalogs with a
governance boundary: **raw PII stays workspace-local; only redacted data reaches the shared metastore.**

This is the serving-first successor to `../vllm/pii_detection_profiling.ipynb`, which ran vLLM in-process
on a classic A100 cluster — not possible on the serverless-only `shm-skunkworks` workspace. Here every
model runs as a serving endpoint and the notebooks are the client.

## The 3-model sweep

| key | model | serving GPU | notes |
|-----|-------|-------------|-------|
| `qwen3-4b` | `Qwen/Qwen3-4B` | `GPU_MEDIUM` (A10) | small base |
| `qwen3-4b-lora` | Qwen3-4B + LoRA (merged) | `GPU_LARGE` (A100) | fine-tuned on the gretel PII data |
| `gemma3-27b` | `google/gemma-3-27b-it` | `GPU_LARGE` (A100) | strong, multilingual; **license-gated** |

The story: does a small model, LoRA-tuned on the task, close the quality gap with the 27B model at a
fraction of the serving cost? `04_compare_models` answers it (F1 vs tokens/s vs latency).

## Data flow (governance)

```
gretel dataset ──▶ shm_skunkworks_catalog.pii.raw_documents      (raw PII — workspace-local, restricted)
                          │
                   serving endpoint (guided-JSON extraction)
                          │
                          ▼
                   shm_catalog.pii.redacted_documents            (masked text + salt IDs only — shared metastore)
```

`redacted_documents` never contains raw values: PII is replaced by placeholders (`PERSON_001`, `SSN_001`,
…) and the entity metadata stores only `entity_type` + `salt_id`. Quality/throughput metrics land in
`shm_skunkworks_catalog.pii.scrub_results` and MLflow (workspace-side).

## Prerequisites

1. **Custom LLM Serving** enabled: *Admin Settings → Previews → Custom LLM Serving → On*.
2. **HF token secret** (Gemma is gated — accept its license on HuggingFace first):
   ```
   databricks secrets create-scope shm
   databricks secrets put-secret shm hf_token
   ```
3. **Serverless GPU** available for the LoRA training step (A10 or larger).
4. `GPU_LARGE` (A100) available for Model Serving — this is confirmed on the first `gemma3-27b` /
   `qwen3-4b-lora` deploy. If A100 serving isn't provisioned, request it (#model-serving) or drop those
   models to `GPU_MEDIUM` with a smaller/quantized model.

## Run order

| # | notebook | what it does | compute |
|---|----------|--------------|---------|
| 00 | `00_setup_and_data.py` | schemas + volume, land raw PII, cache base weights | serverless |
| 01 | `01_lora_finetune_qwen.py` | LoRA fine-tune Qwen3-4B, merge, snapshot to volume | serverless **GPU** |
| 02 | `02_deploy_endpoints.py` | deploy one model via vLLM entrypoint (`model_key` widget) | serverless **GPU** (A10) — see note |
| 03 | `03_scrub_and_redact.py` | scrub raw docs, write redacted to shared catalog, log metrics | serverless |
| 04 | `04_compare_models.py` | side-by-side F1 / throughput / latency | serverless |

Run `02` and `03` once per model (`model_key` ∈ `qwen3-4b`, `qwen3-4b-lora`, `gemma3-27b`). To
run the whole DAG headless, build a multi-task Job from these notebooks in the Jobs UI (the
GPU tasks need the A10/A100 accelerator set in each task's Environment), or `databricks jobs
submit` against them.

## Shared code

- `config.py` — catalogs, tables, model lineup, endpoints, workload types, pinned deps.
- `pii_common.py` — the Pydantic output schema, few-shot system prompt, ground-truth parsing, masking,
  scoring, and the guided-JSON endpoint call. Reused by every notebook (and lifted from the original
  profiling notebook so behavior matches).

## Notes

- **`02` must run on a serverless GPU node (A10).** `env_pack` snapshots the notebook's Python env for
  the GPU serving container; packing vLLM/torch on a CPU node produces an env that crashes the server
  on startup (`exitCode=1`). A10 is enough to *pack* the env for any model; it also *loads* Qwen-4B for
  the smoke test, but not Gemma-27B (so `deploy_gemma` uses `smoke_test=false` — still on A10 for the pack).
- **Scale-to-zero** is requested via `config.SCALE_TO_ZERO_ENABLED=True`. Confirmed **accepted** on the
  entrypoint route on this workspace (2026-09; earlier backends rejected it). `02` still falls back to
  fixed concurrency if a future backend refuses, reporting which mode took (exit JSON `scale_to_zero`).
- **GPU endpoint spin-up is slow** and can exceed the SDK's wait while still `DEPLOYMENT_CREATING` (not an
  error) — `02` catches the timeout and tells you to poll the UI.
- **Structured output** uses vLLM guided decoding via `extra_body={"guided_json": <schema>}` on the
  OpenAI-compatible endpoint — no post-hoc JSON repair needed.
