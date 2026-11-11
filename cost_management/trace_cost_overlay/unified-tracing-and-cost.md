# Unified tracing and LLM cost: how cost reaches a span, what to enable, and how this overlay helps

This note explains how trace cost reaches a span. It tells you what to enable, why
cost can be wrong or absent, and how the overlay in this folder corrects it. The
note uses the MLflow and Databricks docs cited at the end. It also uses live
inspection of two workspaces: the `uaigw_fe` gateway traces and the
`shm_skunkworks_catalog.forecast_demo` MLflow application traces.

## 1. Two kinds of "unified tracing". Do not confuse them.

Both kinds write OpenTelemetry (OTel) spans into Unity Catalog tables. The table
names follow the pattern `<catalog>.<schema>.<prefix>_otel_spans` (and `_otel_logs`,
`_otel_metrics`, `_otel_annotations`). The two kinds are different features. They
record cost differently.

| | MLflow application tracing | AI Gateway unified tracing |
| --- | --- | --- |
| Traces | your instrumented app or agent | all Unity Gateway traffic in the metastore |
| Enabled by | a per-experiment `trace_location` | a one-time metastore-admin setup |
| Cost on the span | yes (`mlflow.llm.cost`), if the conditions are met | no cost attribute |
| Token usage | `mlflow.chat.tokenUsage` | `mlflow.chat.tokenUsage` and `gen_ai.usage.*` |
| Trace-level cost | `mlflow.trace.cost` in `_trace_metadata` | none (no `_trace_metadata` table) |

Neither kind gives a cost that matches the customer rate. The MLflow cost uses a
list rate that can be wrong. The gateway trace has no cost at all. The correction
prices from the Databricks cost tables, not from the span.

## 2. How cost reaches a span (MLflow application tracing)

MLflow records token usage and an estimated USD cost on each LLM-call span. MLflow
then aggregates both to the trace.

- Token usage is the span attribute `mlflow.chat.tokenUsage`.
- Cost is the span attribute `mlflow.llm.cost` (`input_cost`, `output_cost`,
  `total_cost`).
- Model and provider identity are `mlflow.llm.model` and `mlflow.llm.provider`.

MLflow does not price per DBU. MLflow reads a per-token list price from the LiteLLM
price table and multiplies by the token counts. This has two problems. The list
price can differ from the customer contract and region. The per-token model also
ignores provisioned-throughput cost, which pays for reserved capacity, not tokens.

## 3. What to enable to get cost

For MLflow application tracing (to get `mlflow.llm.cost`):

1. Use the correct MLflow version. Token tracking needs 3.2.0 or later. Cost
   tracking needs 3.10.0 or later. The Unity Catalog trace location needs 3.14.0 or
   later.
2. Compute cost on the client. Install LiteLLM in the application, or set the
   `mlflow.llm.cost` attribute on the span yourself.
3. Make the model name on the span match the price table. A custom endpoint alias
   gives a $0.00 cost.
4. Bind the experiment to a Unity Catalog trace location at creation time.

For AI Gateway unified tracing:

1. An account admin enables the Enhanced Unity Gateway beta.
2. A metastore admin opens AI Gateway > Govern > Traces > Set up tracing.
3. The admin selects a catalog and schema, then saves.

This captures tokens but records no cost attribute.

## 4. The authoritative cost tables

Databricks already computes the real cost. The overlay reads these tables:

| Table | Role |
| --- | --- |
| `system.ai_gateway.external_model_spend` | external-model cost, already in USD |
| `system.ai_gateway.usage` | per-request tokens, `request_id`, cache detail |
| `system.billing.usage` with `system.billing.list_prices` | Databricks-hosted DBUs priced to USD |
| `system.serving.endpoint_usage` and `served_entities` | serving token and entity detail |

Two properties make this the right source. `list_prices` holds the region-specific
SKU price, so the cost is region-correct with no config. `list_prices` is also SCD2:
each row has a `price_start_time` and `price_end_time`. The overlay joins on the
usage time window, so each day uses the price that was in effect that day. Price
changes over time need no extra work.

## 5. How this overlay helps

The overlay is a read-only set of Unity Catalog views. It never writes to the
MLflow trace tables. It prices in four steps:

1. Compute authoritative daily USD per model. Use `billing.usage` times
   `list_prices` for Databricks-hosted models. Use `external_model_spend` for
   external models.
2. Compute gateway billable tokens per model per day.
3. Divide USD by tokens to get an effective rate per model per day.
4. Price each request or span by its tokens. Sum the result per trace.

The overlay attributes cost by token share, so the total reconciles to billing. A
model and day with billing but no gateway traffic shows as an unattributed gap. That
gap is the non-gateway usage.

The overlay links traces to cost in two modes:

- Gateway mode. Link each span to a gateway request by `databricks.request_id`. Sum
  the request cost per trace.
- MLflow-span mode. Reprice each LLM span's own tokens by the model rate. Keep the
  original `mlflow.llm.cost` for comparison.

The overlay handles the parts that make a naive query wrong:

- Model identity. It resolves the model from the gateway, MLflow, or billing
  attributes. It strips `system.ai.` and `databricks-` prefixes so the names match.
- Cache reads. The gateway `input_tokens` include cache reads. Serving billing is on
  billable tokens. The overlay prices Databricks-hosted requests on billable tokens
  (`input - cache_read + output`), the same basis as the rate.
- Billing basis. For a provisioned endpoint, the cost pays for reserved capacity.
  The per-token rate is then high and is not a token list price. The overlay flags
  each model and day as `pay_per_token` or `provisioned_or_sparse`.

## 6. The exit. When to delete this overlay.

This overlay is interim. Delete it when MLflow trace cost prices from the platform
instead of the LiteLLM list table. That is, delete it when Gateway/FMAPI pricing (or
a `system`-table cost source) feeds the trace cost directly, so the recorded
`mlflow.llm.cost` already reflects the customer rate. The teardown steps are in
`README.md`.

## Sources

- MLflow — Token usage and cost tracking:
  https://mlflow.org/docs/latest/genai/tracing/token-usage-cost/index.html
- Databricks — MLflow traces in Unity Catalog:
  https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
- Databricks — AI Gateway unified trace table (Beta):
  https://docs.databricks.com/aws/en/ai-gateway/unified-trace-table
- Databricks — AI Gateway usage and external model spend system tables:
  https://docs.databricks.com/aws/en/admin/system-tables/
