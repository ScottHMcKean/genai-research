# AI Runtime & Fine-Tuning — Claims Triage

**Section 4 (Fine Tuning & AIR, incl. Ray).** Showcases the Databricks **AI
Runtime**: distributed AI with **Ray**, a real **LoRA fine-tune** of a small model, and
**MLflow + Unity Catalog** for tracking, registration, and serving — all on the shared
insurance-and-claims spine.

The task throughout: **claims triage** — predict claim `severity` (Low / Medium / High)
from the free-text adjuster `note`.

> **Talk-track material:** see [`REFERENCES.md`](REFERENCES.md) — public blogs/docs, Ray vs
> Spark positioning, FSI use-cases, and the AI Runtime compliance caveat.

## Notebooks (run in order)

| # | Notebook | What it shows | Compute |
|---|----------|---------------|---------|
| 00 | `00_setup.py` | Idempotent. Builds a stratified `triage_train` / `triage_test` split from the shared `claim_notes`. Does **not** regenerate raw data. | Serverless (env 4/5) |
| 01 | `01_ray_ai_runtime.py` | **Ray** fans out zero-shot LLM triage over many notes against the Foundation Model API — driver-local on serverless, multi-node on a classic cluster. Includes a **serial-vs-Ray speedup** comparison and a **Ray Data `map_batches`** pipeline (Ray Core *and* Ray Data). Throughput + accuracy + speedup logged. | Serverless **env 5** (Ray) |
| 02 | `02_finetune_lora.py` | **Real LoRA fine-tune** of `distilbert-base-uncased` → 3-class severity classifier. Logged to MLflow, registered in **Unity Catalog** (`claims_demo.claims_triage_model`). | **Serverless GPU** (A10, AI Runtime); auto **CPU fallback** |
| 03 | `03_serve_and_eval.py` | Fine-tuned vs **zero-shot LLM** on the same test set (accuracy / macro-F1); optional Model Serving deploy. | Serverless (env 4/5) |

## The talk track

For a bank/insurer, most AI at scale is **narrow, high-volume, repetitive** — triage,
routing, classification, extraction. On those tasks a **small fine-tuned model** typically
**matches or beats a frontier LLM**, at **orders-of-magnitude lower cost and latency**, and
runs on-platform under the same governance. Reserve the big LLM for open-ended reasoning.

- **Ray** removes the single-driver bottleneck for distributed inference/featurization
  without leaving Databricks.
- **LoRA** makes fine-tuning cheap: only tiny adapter matrices train; the base stays frozen;
  the artifact is small.
- **MLflow + Unity Catalog** give you tracked runs, a governed model registry, lineage, and
  one-click serving — the MLOps loop a regulated FI needs.

## Prerequisites

- Shared spine built: run `../fins_data/generate_data.py` once (creates
  `shm_skunkworks_catalog.claims_demo.claim_notes`).
- Foundation Model API access (uses `databricks-claude-sonnet-4-5`, pay-per-token).

## GPU / cost notes

- The fine-tune (02) needs a GPU. The DAB job uses a **single-node `g5.xlarge` (A10G)** job
  cluster — a LoRA run over a few hundred short notes finishes in **minutes**.
- **Serverless GPU**: if enabled on the workspace, point task 02 at `serverless_v5` instead
  of the GPU job cluster (see the comment in `resources_ai_runtime_job.yml`).
- **No GPU at all?** The notebook auto-detects CUDA and **falls back to CPU** — DistilBERT
  LoRA on this small dataset still trains on CPU in a few minutes, so the demo never blocks.
- Ray (01) and eval (03) are pay-per-token FMAPI calls only — cents.
- Serving (04, in notebook 03) is **off by default** (`DEPLOY=False`) so nothing provisions
  unattended.

## Run it

1. Run `fins_data/generate_data.py` once (builds the common data).
2. Open this folder and run `00 → 03` in order. `02` uses **serverless GPU** (A10) — pick
   the GPU accelerator in the notebook's Environment panel, or use the DAB job below which
   requests it automatically.

Optional — deploy from the repo root as a Job instead:

```bash
databricks bundle deploy -t dev
databricks bundle run -t dev ai_runtime_job
```
