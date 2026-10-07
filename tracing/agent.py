"""Monte Carlo forecasting chat agent — an MLflow `ResponsesAgent` that lets a
forecaster converse with a simulation.

The LLM decides when to run a Monte Carlo simulation; we execute it locally, feed
the percentiles back, and let the model explain them in plain language. This is a
deliberately small **custom** agent so the tracing story is the whole point:

Tracing (the four things this demo shows):
- (a) **Framework autolog + a manual tool span.** `mlflow.openai.autolog()` captures
      every LLM call automatically; `@mlflow.trace(span_type=TOOL)` on
      `run_monte_carlo` gives the simulation its own named span with numeric
      inputs/outputs. `predict` is the root `AGENT` span, so one turn = one trace.
- (c) **Users + sessions.** `predict` reads `custom_inputs={"user","session_id"}`
      and stamps them on the trace as `mlflow.trace.user` / `mlflow.trace.session`,
      so turns group into conversations and every trace is attributable to a person.
      The app passes the *authenticated end user* here (not the app service
      principal) — which is what makes the governance in (d) enforceable.

Kept standalone so `mlflow.pyfunc.log_model(code_paths=["config.py"])` packages it,
and so the Databricks App in `app/` can import `AGENT` directly.
"""
import json
import uuid
from typing import Any, Generator

import numpy as np

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from config import CHAT_MODEL as LLM_ENDPOINT

SYSTEM_PROMPT = (
    "You are a forecasting assistant for a team of planners. When the user asks about "
    "a projection, likelihood, downside, or 'what if', call the run_monte_carlo tool "
    "to simulate it rather than guessing. Then explain the result in plain language: "
    "lead with the median (p50), give the p10-p90 range as the realistic spread, and "
    "state any probability the user asked about. Keep numbers rounded and readable. "
    "If the user has not given enough detail to parameterise a simulation, ask one "
    "concise clarifying question instead of calling the tool."
)
MAX_TURNS = 4  # LLM<->tool round-trips before forcing a final answer

# Tool schema advertised to the LLM (Chat Completions function shape).
MONTE_CARLO_TOOL = {
    "type": "function",
    "function": {
        "name": "run_monte_carlo",
        "description": (
            "Simulate a value that compounds over several periods with uncertain "
            "per-period growth, and return the distribution of the final value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_value": {"type": "number", "description": "Starting value at period 0."},
                "growth_mean": {"type": "number",
                                "description": "Mean per-period growth rate, e.g. 0.03 for 3%."},
                "growth_std": {"type": "number",
                               "description": "Std dev of per-period growth rate, e.g. 0.05."},
                "periods": {"type": "integer", "description": "Number of periods to project."},
                "threshold": {"type": ["number", "null"],
                              "description": "Optional target; returns P(final >= threshold)."},
                "n_sims": {"type": "integer",
                           "description": "Number of simulation paths (default 20000)."},
            },
            "required": ["start_value", "growth_mean", "growth_std", "periods"],
        },
    },
}


@mlflow.trace(span_type=SpanType.TOOL, name="monte_carlo_simulation")
def run_monte_carlo(
    start_value: float,
    growth_mean: float,
    growth_std: float,
    periods: int,
    threshold: float | None = None,
    n_sims: int = 20_000,
) -> dict:
    """Monte Carlo projection of a compounding value. Seeded for reproducible demos.

    Decorated with `@mlflow.trace`, so every call is its own TOOL span carrying the
    exact parameters the LLM chose and the percentiles returned — the span a
    reviewer or an eval scorer would inspect.
    """
    rng = np.random.default_rng(42)
    # Draw per-period growth, compound to a terminal value per path.
    growth = rng.normal(growth_mean, growth_std, size=(int(n_sims), int(periods)))
    finals = float(start_value) * np.prod(1.0 + growth, axis=1)

    p10, p50, p90 = (float(x) for x in np.percentile(finals, [10, 50, 90]))
    result = {
        "start_value": float(start_value),
        "periods": int(periods),
        "n_sims": int(n_sims),
        "p10": round(p10, 2),
        "p50": round(p50, 2),
        "p90": round(p90, 2),
        "mean": round(float(finals.mean()), 2),
    }
    if threshold is not None:
        result["threshold"] = float(threshold)
        result["prob_at_or_above_threshold"] = round(float((finals >= threshold).mean()), 4)
    return result


class ForecastChatAgent(ResponsesAgent):
    def _llm(self, messages: list[dict], tools: list[dict] | None):
        from databricks_openai import DatabricksOpenAI

        kwargs: dict[str, Any] = {"model": LLM_ENDPOINT, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        return DatabricksOpenAI().chat.completions.create(**kwargs)

    def _to_chat(self, msg: dict) -> list[dict]:
        """Convert a Responses-API item into Chat-Completions messages for the LLM."""
        t = msg.get("type")
        if t == "function_call":
            return [{
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": msg["call_id"], "type": "function",
                    "function": {"name": msg["name"], "arguments": msg["arguments"]},
                }],
            }]
        if t == "function_call_output":
            return [{"role": "tool", "content": msg["output"], "tool_call_id": msg["call_id"]}]
        if t == "message" and isinstance(msg.get("content"), list):
            return [{"role": msg["role"],
                     "content": "".join(c.get("text", "") for c in msg["content"])}]
        return [{k: v for k, v in msg.items()
                 if k in ("role", "content", "name", "tool_calls", "tool_call_id")}]

    def _run(self, request: ResponsesAgentRequest) -> Generator[dict, None, None]:
        """Tool-calling loop, yielding Responses-API output items as they occur."""
        history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for inp in request.input:
            history.append(inp.model_dump() if hasattr(inp, "model_dump") else dict(inp))

        for _ in range(MAX_TURNS):
            flat: list[dict] = []
            for m in history:
                flat.extend(self._to_chat(m))
            choice = self._llm(flat, [MONTE_CARLO_TOOL]).choices[0].message
            tool_calls = choice.tool_calls or []

            if not tool_calls:
                yield self.create_text_output_item(text=choice.content or "", id=uuid.uuid4().hex)
                return

            for tc in tool_calls:
                call_id, name, args_json = tc.id, tc.function.name, tc.function.arguments
                yield self.create_function_call_item(
                    id=uuid.uuid4().hex, call_id=call_id, name=name, arguments=args_json)
                history.append({"type": "function_call", "call_id": call_id,
                                "name": name, "arguments": args_json})
                try:
                    output = json.dumps(run_monte_carlo(**json.loads(args_json or "{}")))
                except Exception as e:  # surfaced to the model + UI
                    output = json.dumps({"error": f"{type(e).__name__}: {e}"})
                yield self.create_function_call_output_item(call_id=call_id, output=output)
                history.append({"type": "function_call_output", "call_id": call_id, "output": output})

        # Fell through MAX_TURNS: force a final answer without tools.
        flat = []
        for m in history:
            flat.extend(self._to_chat(m))
        final = self._llm(flat, None).choices[0].message.content or ""
        yield self.create_text_output_item(text=final, id=uuid.uuid4().hex)

    def _tag_trace(self, request: ResponsesAgentRequest) -> None:
        """Stamp user + session onto the current trace from custom_inputs.

        Session metadata is immutable once set, so do this before any child span.
        The app passes the authenticated end-user's email as `user` — that identity
        is what the secure view in `01_custom_agent_tracing.py` keys on."""
        ci = request.custom_inputs or {}
        meta = {}
        if ci.get("user"):
            meta["mlflow.trace.user"] = str(ci["user"])
        if ci.get("session_id"):
            meta["mlflow.trace.session"] = str(ci["session_id"])
        if meta:
            mlflow.update_current_trace(metadata=meta)

    # NOTE: do not decorate predict/predict_stream with @mlflow.trace — MLflow already
    # auto-traces a ResponsesAgent's predict as the root span. Decorating it too creates
    # a second root (two traces per turn). We just tag the auto-created trace here.
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        self._tag_trace(request)
        return ResponsesAgentResponse(output=list(self._run(request)))

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # Emit each item (tool call, tool output, final message) as produced so the
        # chat UI can render the simulation step live.
        self._tag_trace(request)
        for item in self._run(request):
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)


mlflow.openai.autolog()  # (a) auto-trace every LLM call in this agent
AGENT = ForecastChatAgent()
mlflow.models.set_model(AGENT)
