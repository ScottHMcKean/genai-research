# Interim corrected-trace-cost overlay

A **read-only** overlay that recomputes LLM cost from MLflow/OTel traces stored in
Unity Catalog, correcting for a pricing mismatch in the current product. Delete it
when the product fix lands.

## The problem

MLflow trace cost is estimated at an AWS/GCP list rate (~**$0.07/DBU**) instead of
the customer's Azure North Europe rate (**$0.105/DBU** list, ~**$0.092/DBU**
effective). This overlay reads the trace spans and writes corrected cost into its
own objects. It never modifies the MLflow-managed trace tables.

## Hard rules honoured

- No writes to `<prefix>_otel_spans | _otel_logs | _otel_metrics | _otel_annotations`
  or `<prefix>_trace_metadata | _trace_unified`. The overlay only reads them.
- Rates live in `trace_cost_rate_config`, never in code.
- All objects live in `output_catalog.output_schema` and are safe to drop.

## What the discovery found (uaigw test bed)

The test workspace uses **AI Gateway unified tracing**, which differs from the
customer's shape in one decisive way:

| Aspect | Customer (expected) | uaigw test bed (actual) |
| --- | --- | --- |
| Per-span cost | `mlflow.llm.cost` present | **absent** — no cost anywhere |
| Token usage | `mlflow.chat.tokenUsage` | present, plus `gen_ai.usage.*` |
| `tokenUsage` encoding | nested object | **JSON string** (needs `parse_json(...::string)`) |
| Model on the kept span | on the LLM span | blank; recovered from child span / endpoint |
| Spans per call | — | **two** (parent `LLM` + child `chat`), identical tokens |
| Cache read tokens | counted in cost | `cache_read_input_tokens` inside `tokenUsage`, can be ~all input |
| Trace-level cost | `mlflow.trace.cost` | none (no `_trace_metadata` table) |

Because there is no `mlflow.llm.cost` to multiply, the overlay prices from tokens on
this workspace — which is exactly the fallback the spec calls for when the MLflow
cost does not imply a flat $/DBU.

## Two pricing engines (chosen per span)

1. **`dbu_multiplier`** — span has `mlflow.llm.cost`.
   `corrected = mlflow_total_cost × (target_rate / mlflow_assumed_rate)`,
   where `target_rate` is `azure_list_rate` or `effective_rate` per `pricing_mode`.
   With the seeded rates this is **1.5×** (list) or **~1.3143×** (effective).
   This is the customer's path.
2. **`token`** — span has only token usage.
   `corrected = billable_input/1e6 × input_rate + cache_read/1e6 × cache_rate +
   output/1e6 × output_rate`, with `billable_input = input_tokens − cache_read`.
   This is the uaigw path.

Unmatched models are **passed through, never dropped**, and flagged
`rate_matched = false` (they fall to the `%` default rate row).

## Dedup

Each gateway call emits a parent `LLM` span and a child `chat` span carrying the
**same** token usage. Summing every span double-counts. Reproducing MLflow's rule
(skip a span if an ancestor carries the same attribute), the trace view keeps only
the **top-most** cost-bearing span per branch (`is_topmost_cost_span`). Verified:
75,247 token spans → 37,624 top-most (exactly half); trace-view total equals the
top-most span total and is half the naive sum.

## Objects created (all in `output_catalog.output_schema`)

| Object | Kind | Purpose |
| --- | --- | --- |
| `trace_cost_settings` | table | runtime `pricing_mode` read by the views |
| `trace_cost_rate_config` | table | rates (seeded) |
| `v_trace_span_cost_corrected` | view | one row per cost-bearing span |
| `v_trace_cost_corrected` | view | one row per trace (deduped) |
| `v_trace_cost_reconciliation` | view | corrected vs `system.billing.usage` |
| `trace_cost_corrected` | table | optional materialization (job target) |

## Run it

Parameters (notebook widgets / bundle variables): `trace_catalog`, `trace_schema`,
`table_prefix`, `output_catalog`, `output_schema`, `pricing_mode`
(`effective` | `azure_list`).

1. **Setup + views + validation + reconciliation** — run
   `src/setup_trace_cost_overlay.py` with the widgets set.
2. **Materialize (optional, daily)** — deploy the bundle and run the job:
   ```bash
   databricks bundle deploy -t dev --profile <PROFILE>
   databricks bundle run materialize_trace_cost -t dev --profile <PROFILE>
   ```
   The job MERGEs `v_trace_cost_corrected` into `trace_cost_corrected` on
   `trace_id`, incrementally, with a 48h watermark lookback on `trace_start_time`.
   It runs on serverless; in `dev` mode the schedule is paused.
3. **Dashboard** — deployed by the bundle
   (`dashboards/trace_cost_overlay.lvdash.json`).

## How to change rates

Edit `trace_cost_rate_config` — no code change:

- **Token rates**: update `input_rate_per_mtok`, `output_rate_per_mtok`,
  `cache_read_rate_per_mtok` (USD per 1M tokens) for the model pattern.
- **DBU multiplier**: update `azure_list_rate` / `effective_rate` /
  `mlflow_assumed_rate`.
- **Precedence**: lower `priority` wins; the `%` row at priority 100 is the default
  and marks spans `rate_matched = false`.
- **Switch mode**: `INSERT OVERWRITE trace_cost_settings VALUES ('pricing_mode','azure_list')`.

> The seeded `token` rates are **PLACEHOLDERS** (fictional test-bed models). Replace
> them with real per-token rates before trusting dollar figures.

## Caveats

- MLflow prices from LiteLLM per-token list prices, not per-DBU. On the skunkworks
  check, `databricks-claude-sonnet-4-5` implied **$3 / M input, $15 / M output** —
  no DBU in the math. Confirm the implied $/DBU before relying on a flat multiplier;
  the setup notebook prints the implied per-token price for `mlflow.llm.cost` spans.
- Cache read tokens are inside `input_tokens` but priced separately. On uaigw,
  `claude-opus-5` was ~92% cache reads; pricing them at full input rate nearly
  doubles the estimate.
- Some spans have tokens but no resolvable model → default rate, `rate_matched=false`.
  On uaigw this was 0 (model always recovered from child or endpoint).
- `mlflow.trace.cost` validation runs only when a `<prefix>_trace_metadata` table
  exists; it is N/A on token-only workspaces.
- Reconciliation on a shared workspace compares against *all* model-serving usage,
  not just traced calls, so the gap is large on a test bed.

## Teardown

Run the teardown cell at the end of `src/setup_trace_cost_overlay.py`, or:

```sql
DROP VIEW  IF EXISTS <out>.v_trace_cost_reconciliation;
DROP VIEW  IF EXISTS <out>.v_trace_cost_corrected;
DROP VIEW  IF EXISTS <out>.v_trace_span_cost_corrected;
DROP TABLE IF EXISTS <out>.trace_cost_corrected;
DROP TABLE IF EXISTS <out>.trace_cost_rate_config;
DROP TABLE IF EXISTS <out>.trace_cost_settings;
DROP SCHEMA IF EXISTS <out>;
```
Then `databricks bundle destroy -t dev` to remove the job and dashboard.
