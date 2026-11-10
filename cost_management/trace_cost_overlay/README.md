# Corrected trace cost overlay

A **read-only** overlay that recomputes LLM trace cost from the **authoritative
Databricks cost tables** instead of MLflow's LiteLLM list estimate. Delete it when
the product fix lands.

## The problem

MLflow trace cost (`mlflow.llm.cost`) is a per-token estimate from the LiteLLM list
table. It can use the wrong region/contract rate (the original "$0.07 vs $0.105"
mismatch) and it ignores provisioned-throughput economics. An earlier version of
this overlay scaled that estimate by a $/DBU ratio — that mixed a per-token base
with a per-DBU base and was wrong. This version prices from source.

## Authoritative sources

| Table | Role |
| --- | --- |
| `system.ai_gateway.external_model_spend` | external-model cost, already in USD, per model/endpoint/hour |
| `system.ai_gateway.usage` | per-request tokens + `request_id` + full cache detail |
| `system.billing.usage` x `system.billing.list_prices` | Databricks-hosted DBU -> USD; **region-correct**; `list_prices` is **SCD2** on the price time window |
| `system.serving.endpoint_usage` / `served_entities` | serving token and entity detail |

Because `list_prices` already carries the region-specific SKU price and is SCD2, the
region mismatch disappears with **no rate config**, and price changes over time are
handled by the usage-time join.

## Method

1. `v_model_cost_daily` — authoritative daily USD per model (billing x list_prices
   for Databricks-hosted, `external_model_spend` for external). Contract discount
   applied here.
2. `v_model_tokens_daily` — gateway billable tokens per model/day (Databricks: ex
   cache-read; external: input+output).
3. `v_model_rate_daily` — effective `usd_per_token = usd / tokens`, with a
   `billing_basis` flag. `v_model_rate_scd2` collapses it into rate periods;
   `v_list_price_history` shows the list-price SCD2 history.
4. `v_request_cost` — each gateway request priced by its tokens x the model rate.
5. `v_trace_cost_corrected` — cost per trace. Two modes (see below).
6. `v_trace_cost_reconciliation` — attributed vs authoritative USD per day.

Allocation by token share makes the total **reconcile to billing by construction**.
A model/day with billing but no gateway traffic appears as an unattributed gap (=
non-gateway usage).

### Trace link modes (`trace_mode` widget)

- **`gateway`** (AI Gateway unified tracing, e.g. uaigw): link spans to requests by
  `databricks.request_id`, then sum `v_request_cost` per trace.
- **`mlflow_span`** (MLflow app tracing, e.g. skunkworks `forecast_demo`): reprice
  each LLM span's own tokens (`mlflow.chat.tokenUsage`) by the model rate. Keeps the
  original `mlflow.llm.cost` for comparison.

## `billing_basis`: pay-per-token vs provisioned

For pay-per-token models the effective rate is a true token price and is stable. For
**provisioned-throughput** endpoints you pay for reserved capacity, so `usd/token` is
high and varies with utilisation — it is **not** a token list price. The rate layer
flags each model/day as `pay_per_token` or `provisioned_or_sparse`. Note MLflow's
`mlflow.llm.cost` **understates** provisioned cost because it prices per token only.

## Objects (all in `output_catalog.output_schema`)

Tables: `trace_cost_rate_contract` (config), `trace_cost_corrected` (materialized).
Views: `v_model_cost_daily`, `v_model_tokens_daily`, `v_model_rate_daily`,
`v_model_rate_scd2`, `v_list_price_history`, `v_request_cost`,
`v_trace_cost_corrected`, `v_trace_cost_reconciliation`.

## Run it

Widgets: `trace_catalog`, `trace_schema`, `table_prefix`, `output_catalog`,
`output_schema`, `trace_mode` (`gateway` | `mlflow_span`).

1. Run `src/setup_trace_cost_overlay.py` with the widgets set. It creates the config,
   the views, and prints the validation (rate x tokens == billing; reconciliation
   percentage).
2. Materialize (optional, daily) via the bundle:
   ```bash
   databricks bundle deploy -t dev --profile <PROFILE>
   databricks bundle run materialize_trace_cost -t dev --profile <PROFILE>
   ```
   The job MERGEs `v_trace_cost_corrected` into `trace_cost_corrected` on
   `trace_id`, incrementally, with a watermark on `trace_date`. Serverless; the
   schedule is paused in `dev`.
3. The AI/BI dashboard ships in the bundle.

## How to change rates

- **Region/list price**: nothing to do — `list_prices` is authoritative and SCD2.
- **Negotiated discount**: add a row to `trace_cost_rate_contract`, e.g.
  `('claude%','northeurope',0.88, DATE'2026-01-01', NULL, 10, '12% off list')`.
  `discount_factor = 1.0` is list. External models already bill actual USD.

## Tested

- **uaigw** (`gateway` mode): 2.09M-span gateway table. Daily reconciliation 98-100%
  on most days; rate x tokens == billing within rounding. External Azure/OpenAI/etc.
  priced directly from `external_model_spend`.
- **skunkworks** (`mlflow_span` mode): `forecast_demo` traces repriced from billing;
  surfaced that the endpoint is provisioned (high `usd/token`), which MLflow's
  per-token estimate missed.

## Caveats

- Requests whose model has no billing attribution that day are flagged
  `rate_matched = false` (not dropped).
- `provisioned_or_sparse` per-token rates reflect capacity, not list price.
- Reconciliation gap = billing with no gateway traffic (non-gateway callers).
- `request_id` is present on ~93% of kept gateway spans; traces without it are not
  linked in `gateway` mode.

## Teardown

Uncomment the teardown cell in `src/setup_trace_cost_overlay.py`, then
`databricks bundle destroy -t dev`.
