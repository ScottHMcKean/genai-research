# Document Intelligence — `ai_parse_document` → `ai_extract` → MLflow Evaluate

A clean, runnable demo of insurance document extraction on Databricks: parse PDFs to a
table, extract structured fields with `ai_extract`, query the **cost**, and **evaluate**
extraction quality against a **hand-labelled benchmark** with `mlflow.genai.evaluate`.

Built for an underwriting use case, but the loop is generic to any document-extraction task.

## Pipeline

```
real public insurance PDFs in a Volume
      │  ai_parse_document          (layout-aware text + OCR)
      ▼
parsed_docs ──► parsed_text         (materialized Delta tables)
      │  ai_extract  (v2.1: per-field citations + confidence)
      ▼
extracted_fields ──► extracted_flat (one column per field + confidence)
      │  mlflow.genai.evaluate      (table-driven: inputs+outputs+expectations → traces + judges)
      ▼
MLflow experiment: field accuracy, coverage recall, LLM judge, traces
```

- **Runs on serverless, environment v5**, `%uv` for installs.
- `ai_extract` works on **already-parsed text** (it can't read PDF bytes) — so
  `ai_parse_document` runs first and we materialize its output. (`ai_extract` /
  `ai_parse_document` / `ai_classify` use Databricks-managed models; only `ai_query` lets
  you pick a model.)
- Evaluation is **table-driven** — no live call, no `spark.sql` in scorers — so the same
  scorers apply whether extraction ran as a batch job or behind a model-serving endpoint.

## Notebooks (run in order)

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_setup_and_data.ipynb` | Schema/volume; downloads **10 real public ACORD/CMS PDFs**; writes the **hand-labelled** `golden_labels` table. |
| 01 | `01_parse_and_cost.ipynb` | `ai_parse_document` → `parsed_docs` / `parsed_text`; queries **`AI_FUNCTIONS` cost** from `system.billing`; throughput. |
| 02 | `02_extract_fields.ipynb` | `ai_extract` (v2.1 citations + confidence) → `extracted_fields` / `extracted_flat`; **confidence-based review queue**. |
| 03 | `03_evaluate_mlflow.ipynb` | Table-driven `mlflow.genai.evaluate` with **tolerant code scorers + a deploy-client LLM judge**. |

Config (catalog/schema/volume) is at the top of each notebook — defaults to
`users.scott_mckean.insurance_pdfs`.

## Data & labels

No clean public *labelled* insurance corpus exists, so the benchmark is **10 real public
documents** (ACORD certificates + a CMS-1500) with **hand labels**. They're genuinely messy
— OCR, multi-column tables, and template placeholders — which makes the eval credible. The
labels encode two non-obvious rules:

- **Placeholders are not values** — `"Insert Insurer name"`, `"Carrier A"` → null.
- **Primary certificate only** for multi-certificate packages.

Real certificates of liability carry **policy + coverage** fields, not property
**underwriting** fields (flood zone, construction, year built, prior claims, hazmat,
sprinklers). Those live on declarations pages / SOVs — add the customer's own dec/SOV docs
with labels and the same loop scores them.

**Fields:** `document_type`, `insurer_name`, `insured_name`, `policy_number`,
`effective_date`, `expiration_date`, `coverage_types`.

## What a run looks like (real numbers)

On the 9–10 public docs, a recent run scored:

| metric | mean |
|---|---|
| field_accuracy | ~0.78 |
| document_type / dates / policy_number | ~0.89 |
| insured_name / insurer_name (code match) | ~0.56 |
| **insurer_name (LLM judge)** | **~0.44** |
| coverage_recall | ~0.85 |

The LLM judge scoring *lower* than the code match is the point: it correctly rejects
template placeholders that a string match lets through. Code scorers are cheap and
deterministic; LLM judges catch semantic/placeholder errors.

## Cost notes

- `ai_parse_document` ≈ **$1–6 / 1,000 pages**; combined parse + extract ≈ **$8.75 / 1k**
  at medium complexity. Bills under **`AI_FUNCTIONS`** in `system.billing.usage`
  (`system.billing` can lag a few hours, so notebook 01 also shows an a-priori estimate).
- No per-query cost controls today — at scale, route with `ai_classify` and only parse docs
  that need layout/OCR.

## References

- Official IDP Solution Accelerator — `databricks-solutions/databricks-blogposts`
  `/2026-04-Intelligent-document-processing` (parse → extract → MLflow evaluate).
- Agentic Underwriting / Bricksurance — `wryszka/agentic-underwriting-v2`.
- Incremental parse DAB — `databricks/bundle-examples/contrib/job_with_ai_parse_document`.
- Docs: [AI Functions](https://docs.databricks.com/aws/en/large-language-models/ai-functions),
  [Intelligent Document Processing](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/intelligent-document-processing),
  [MLflow 3 scorers & judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers).
