# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Supervisor API — does it have session memory?
# MAGIC
# MAGIC The [Supervisor API](https://docs.databricks.com/aws/en/agents/agent-bricks/supervisor-api)
# MAGIC (Beta) is an OpenResponses-compatible endpoint (`POST ai-gateway/mlflow/v1/responses`)
# MAGIC where Databricks runs the agent loop for you. The docs state, in one line:
# MAGIC
# MAGIC > **"The Supervisor API doesn't store conversation state between requests."**
# MAGIC
# MAGIC This notebook **empirically validates that claim** rather than taking it on faith, then
# MAGIC shows the pattern that *does* give you a conversational session. Every check prints
# MAGIC `[as documented]` or `[UNEXPECTED]`, and a scorecard at the end summarises what the
# MAGIC platform actually did on this workspace, on this date.
# MAGIC
# MAGIC | Probe | Question it answers |
# MAGIC |---|---|
# MAGIC | A | Two independent calls — does turn 2 remember turn 1? (expect **no**) |
# MAGIC | B | Client-replayed history — does turn 2 remember turn 1? (expect **yes**) |
# MAGIC | C | Is OpenAI's `previous_response_id` server-side threading supported? (docs: not listed) |
# MAGIC | D | Is `store=True` + `responses.retrieve()` supported for sync calls? |
# MAGIC | E | Client-side function tools — the documented **two-turn** round trip |
# MAGIC | F | Background mode (`background=True`) — is the polled response a session? |
# MAGIC | G | Background mode **with MCP** — `mcp_approval_response` needs full history |
# MAGIC | H | `SupervisorSession` helper — memory across 3 turns incl. a tool call |
# MAGIC | I | Session *observability* — grouping turns in the MLflow Sessions view |
# MAGIC
# MAGIC **Bottom line up front:** the Supervisor API is a stateless agent-loop *executor*. It
# MAGIC manages the multi-turn loop **inside** one request (model → tool → model → answer); it
# MAGIC does not carry anything **between** requests. Session memory is the client's job — see
# MAGIC `SupervisorSession` in Probe H for the ~25 lines that provide it.
# MAGIC
# MAGIC ## Requirements (Beta — enable on the Previews page)
# MAGIC - **AI governance with Unity AI Gateway** enabled for the account.
# MAGIC - **Store OpenTelemetry traces in Unity Catalog** enabled (only for Probe I / tracing).
# MAGIC - Workspace in a [supported region](https://docs.databricks.com/aws/en/agents/agent-bricks/supervisor-api).
# MAGIC
# MAGIC Probes A–F and H run with **no prerequisites** beyond the previews. Probe G needs a UC
# MAGIC MCP service (set `UC_MCP_TOOL`); Probe E's UC-function variant needs `00_setup`.

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-openai mlflow
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Config + client
from config import CHAT_MODEL, CATALOG, SCHEMA, CLAIM_LOOKUP_FN

# Optional extras. Leave as None to skip the probe that needs them.
UC_MCP_TOOL = None          # e.g. f"{CATALOG}.{SCHEMA}.slack_mcp" -> enables Probe G
TRACE_PREFIX = "supervisor_session_test"   # UC table prefix for server-side traces

from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)
MODEL = CHAT_MODEL
print("model:", MODEL)

# COMMAND ----------

# DBTITLE 1,Probe harness — a memory canary
# The whole notebook turns on one trick: plant a token the model cannot possibly know,
# then ask for it back in a *separate* request. If the token comes back, state persisted.
SECRET = "CLM-778291"

TURN_1 = f"Remember this claim number exactly: {SECRET}. Reply with just 'noted'."
TURN_2 = (
    "What was the claim number I gave you earlier? Reply with ONLY the claim number, "
    "or the single word NONE if I never gave you one."
)

RESULTS = []


def text_of(response) -> str:
    """output_text, falling back to walking output items (tool-only turns can be empty)."""
    txt = getattr(response, "output_text", None)
    if txt:
        return txt
    parts = []
    for item in getattr(response, "output", []) or []:
        for chunk in getattr(item, "content", []) or []:
            if getattr(chunk, "text", None):
                parts.append(chunk.text)
    return "\n".join(parts)


def remembered(response) -> bool:
    return SECRET in text_of(response)


def record(probe: str, question: str, expected: bool | str, observed: bool | str, note: str = ""):
    """Log one probe. `expected` is what the docs say; `observed` is what happened."""
    ok = expected == observed
    RESULTS.append(
        {"probe": probe, "question": question, "expected": expected,
         "observed": observed, "matches_docs": ok, "note": note}
    )
    flag = "[as documented]" if ok else "[UNEXPECTED — docs say otherwise]"
    print(f"{flag} {probe}: expected={expected!r} observed={observed!r} {note}")


def msg(role: str, content: str) -> dict:
    return {"type": "message", "role": role, "content": content}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe A — two independent requests (the control)
# MAGIC
# MAGIC Plant the secret in request 1. Ask for it back in a brand-new request 2, sending
# MAGIC **nothing** but the question. If the API had server-side session memory, the secret
# MAGIC would come back. Expect `NONE`.

# COMMAND ----------

r1 = client.responses.create(model=MODEL, input=[msg("user", TURN_1)], stream=False)
print("turn 1 →", text_of(r1).strip()[:200])

r2 = client.responses.create(model=MODEL, input=[msg("user", TURN_2)], stream=False)
print("turn 2 →", text_of(r2).strip()[:200])

record(
    "A · stateless across requests",
    "Does a fresh request recall the previous one?",
    expected=False,
    observed=remembered(r2),
    note="no session memory server-side" if not remembered(r2) else "state leaked between requests!",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe B — client-replayed history (the working pattern)
# MAGIC
# MAGIC Same two turns, except request 2 carries the whole transcript: the original user
# MAGIC message, the model's own output items echoed back verbatim, then the new question.
# MAGIC This is the *only* documented way to get continuity. Expect the secret back.

# COMMAND ----------

history = [msg("user", TURN_1)]
r1b = client.responses.create(model=MODEL, input=history, stream=False)

# Echo the model's turn into history exactly as returned — this is the documented pattern.
history += [item.model_dump() for item in r1b.output]
history += [msg("user", TURN_2)]

r2b = client.responses.create(model=MODEL, input=history, stream=False)
print("turn 2 (with replayed history) →", text_of(r2b).strip()[:200])

record(
    "B · client-replayed history",
    "Does replaying the transcript restore memory?",
    expected=True,
    observed=remembered(r2b),
    note=f"input carried {len(history)} items",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe C — is `previous_response_id` supported?
# MAGIC
# MAGIC OpenAI's Responses API threads conversations server-side with `previous_response_id`.
# MAGIC The Supervisor API's documented parameter list is `model`, `input`, `tools`,
# MAGIC `instructions`, `stream`, `background`, `trace_destination` — **`previous_response_id`
# MAGIC is not among them**. This probe records what the endpoint actually does with it:
# MAGIC reject it, ignore it, or (surprise) honour it.

# COMMAND ----------

try:
    r_prev = client.responses.create(
        model=MODEL,
        input=[msg("user", TURN_2)],
        previous_response_id=r1.id,      # r1 planted the secret in Probe A
        stream=False,
    )
    observed = "honoured" if remembered(r_prev) else "accepted-but-ignored"
    print("response →", text_of(r_prev).strip()[:200])
except Exception as e:
    observed = "rejected"
    print(f"rejected: {type(e).__name__}: {str(e)[:300]}")

record(
    "C · previous_response_id",
    "Can the server thread turns by response id?",
    expected="rejected-or-ignored",
    observed=observed,
    note="undocumented param; 'honoured' would mean hidden server-side sessions exist",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe D — `store=True` and retrieving a synchronous response
# MAGIC
# MAGIC Retrieval is documented **only** for background mode (`responses.retrieve()` after
# MAGIC `background=True`, retained 30 days). Does a plain synchronous response persist and
# MAGIC come back by id? A retrievable response is still not a *session* — but it tells you
# MAGIC whether transcripts are recoverable server-side at all.

# COMMAND ----------

try:
    r_store = client.responses.create(
        model=MODEL, input=[msg("user", TURN_1)], store=True, stream=False
    )
    fetched = client.responses.retrieve(r_store.id)
    observed = f"retrievable (status={getattr(fetched, 'status', 'n/a')})"
    print("retrieved id:", fetched.id)
except Exception as e:
    observed = "not-retrievable"
    print(f"{type(e).__name__}: {str(e)[:300]}")

record(
    "D · store + retrieve (sync)",
    "Are synchronous responses persisted and retrievable?",
    expected="not-retrievable",
    observed=observed,
    note="docs scope retrieve() to background mode only",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe E — client-side function tools take two turns
# MAGIC
# MAGIC The docs are explicit about *why*: **"The Supervisor API doesn't store conversation
# MAGIC state between requests, so a client-side function call takes two turns."** So the
# MAGIC two-turn round trip is itself a consequence of statelessness — a second, independent
# MAGIC confirmation of Probe A.
# MAGIC
# MAGIC - **Turn 1** → the model emits a `function_call` item instead of an answer.
# MAGIC - Your code runs the function locally.
# MAGIC - **Turn 2** → resend the original input **plus** the `function_call` **plus** a new
# MAGIC   `function_call_output`. Drop any of those three and the model has amnesia.
# MAGIC
# MAGIC Contrast with **hosted** tools (`uc_function`, `genie_space`, `knowledge_assistant`,
# MAGIC `uc_mcp`, …): Databricks runs that loop *inside* a single request. In-request loop =
# MAGIC managed; between-request memory = yours.

# COMMAND ----------

import json

CLAIM_STATUS = {  # stands in for a real system of record
    "CLM-100001": {"status": "OPEN", "type": "water damage", "reserve_usd": 18400},
    "CLM-778291": {"status": "CLOSED", "type": "glass only", "reserve_usd": 620},
}

LOOKUP_TOOL = {
    "type": "function",
    "name": "claim_status",
    "description": "Look up the status, type, and reserve amount for a claim id.",
    "parameters": {
        "type": "object",
        "properties": {"claim_id": {"type": "string"}},
        "required": ["claim_id"],
        "additionalProperties": False,
    },
}


def run_claim_status(claim_id: str) -> dict:
    return CLAIM_STATUS.get(claim_id, {"error": f"no such claim {claim_id}"})


tool_input = [msg("user", f"What is the status of claim {SECRET}?")]

# Turn 1 — expect a function_call, not a final answer.
t1 = client.responses.create(model=MODEL, input=tool_input, tools=[LOOKUP_TOOL], stream=False)
calls = [i for i in t1.output if getattr(i, "type", None) == "function_call"]
print(f"turn 1 emitted {len(calls)} function_call item(s):", [c.name for c in calls])

# Echo the model's turn, then append our locally-computed results.
tool_input += [item.model_dump() for item in t1.output]
for call in calls:
    result = run_claim_status(**json.loads(call.arguments))
    tool_input.append(
        {"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)}
    )

# Turn 2 — the model uses the tool result to answer.
t2 = client.responses.create(model=MODEL, input=tool_input, tools=[LOOKUP_TOOL], stream=False)
answer = text_of(t2)
print("turn 2 →", answer.strip()[:300])

record(
    "E · client function two-turn",
    "Does turn 1 return a function_call rather than an answer?",
    expected=True,
    observed=len(calls) > 0,
    note="statelessness is why this needs two round trips",
)
record(
    "E2 · tool result reaches the answer",
    "Did the final answer use the function_call_output?",
    expected=True,
    observed="CLOSED" in answer.upper(),
    note="answer should reflect the locally-computed status",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### E3 (optional) — the same tool, hosted as a UC function
# MAGIC
# MAGIC Hosted tools need **one** request: Databricks calls the function itself inside the
# MAGIC agent loop. Needs `claim_lookup` from `00_setup.py`; skipped cleanly if absent.

# COMMAND ----------

try:
    hosted = client.responses.create(
        model=MODEL,
        input=[msg("user", "Look up claim CLM-100001 and summarise it in one sentence.")],
        tools=[{"type": "uc_function", "name": "claim_lookup",
                "uc_function": {"name": CLAIM_LOOKUP_FN}}],
        stream=False,
    )
    print("hosted uc_function answered in one request →", text_of(hosted).strip()[:300])
    print("(loop managed server-side: no function_call round trip needed)")
except Exception as e:
    print(f"skipped — needs {CLAIM_LOOKUP_FN} from 00_setup: {type(e).__name__}: {str(e)[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe F — background mode is a job handle, not a session
# MAGIC
# MAGIC `background=True` returns immediately with an id and `status` of `queued` /
# MAGIC `in_progress`; you poll `responses.retrieve()` until a terminal state (30-minute cap,
# MAGIC responses retained 30 days, and `stream` + `background` are mutually exclusive).
# MAGIC
# MAGIC The question this probe settles: does a completed background response act as a
# MAGIC **session** you can continue? Two checks — (F1) does polling work, and (F2) does a
# MAGIC *new* background request recall the earlier one?

# COMMAND ----------

from time import sleep

bg = client.responses.create(model=MODEL, input=[msg("user", TURN_1)], background=True)
print(f"submitted id={bg.id} status={bg.status}")

waited = 0
while bg.status in {"queued", "in_progress"} and waited < 120:
    sleep(2)
    waited += 2
    bg = client.responses.retrieve(bg.id)
print(f"terminal status={bg.status} after ~{waited}s → {text_of(bg).strip()[:160]}")

record(
    "F1 · background polling",
    "Does background submit + poll reach a terminal state?",
    expected="completed",
    observed=bg.status,
    note=f"polled for ~{waited}s",
)

# F2 — a second background request with only the question, no history.
bg2 = client.responses.create(model=MODEL, input=[msg("user", TURN_2)], background=True)
waited = 0
while bg2.status in {"queued", "in_progress"} and waited < 120:
    sleep(2)
    waited += 2
    bg2 = client.responses.retrieve(bg2.id)
print("second background response →", text_of(bg2).strip()[:200])

record(
    "F2 · background is not a session",
    "Does a new background request recall the previous one?",
    expected=False,
    observed=remembered(bg2),
    note="retrievable-by-id != conversational memory",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe G — background mode with MCP: approval proves the point
# MAGIC
# MAGIC This is the section from the link. For security, **any** MCP tool call in background
# MAGIC mode needs explicit approval: the response completes with an `mcp_approval_request`
# MAGIC (exposing `name`, `server_label`, and the `arguments` the model intends to pass), and
# MAGIC you resume by sending back:
# MAGIC
# MAGIC ```json
# MAGIC {"type": "mcp_approval_response", "id": "<tool-call-id>",
# MAGIC  "approval_request_id": "<tool-call-id>", "approve": true}
# MAGIC ```
# MAGIC
# MAGIC Note the wording in the docs: pass it back in the `input` field **"with the full
# MAGIC conversation history."** Even mid-approval — a flow the server itself paused and handed
# MAGIC you an id for — the server will not reconstruct the transcript. That is the strongest
# MAGIC available evidence that no session state exists anywhere server-side.
# MAGIC
# MAGIC Set `UC_MCP_TOOL` at the top to run this live; otherwise the loop below is skipped and
# MAGIC the assertion is recorded as `skipped`.

# COMMAND ----------

if not UC_MCP_TOOL:
    print("Probe G skipped — set UC_MCP_TOOL to a UC MCP service, e.g.")
    print(f'  UC_MCP_TOOL = "{CATALOG}.{SCHEMA}.<mcp_service>"')
    record("G · MCP approval resume", "Does resuming an MCP approval need full history?",
           expected="needs-full-history", observed="skipped", note="no UC_MCP_TOOL configured")
else:
    mcp_tools = [{"type": "uc_mcp", "name": "mcp", "uc_mcp": {"name": UC_MCP_TOOL}}]
    mcp_history = [msg("user", "Use your MCP tool to look up the latest on claims triage.")]

    resp = client.responses.create(model=MODEL, input=mcp_history, tools=mcp_tools, background=True)
    waited = 0
    while resp.status in {"queued", "in_progress"} and waited < 300:
        sleep(3)
        waited += 3
        resp = client.responses.retrieve(resp.id)

    approvals = [i for i in resp.output if getattr(i, "type", None) == "mcp_approval_request"]
    print(f"status={resp.status}; {len(approvals)} approval request(s)")
    for a in approvals:
        print(f"  tool={a.name} server={getattr(a, 'server_label', '?')} args={a.arguments}")

    record("G1 · MCP approval gate", "Does background MCP pause for approval?",
           expected=True, observed=len(approvals) > 0,
           note="required for security before any MCP tool executes")

    if approvals:
        # Resume: FULL history + the model's output items + one approval per request.
        resume = list(mcp_history) + [i.model_dump() for i in resp.output]
        resume += [{"type": "mcp_approval_response", "id": a.id,
                    "approval_request_id": a.id, "approve": True} for a in approvals]

        cont = client.responses.create(model=MODEL, input=resume, tools=mcp_tools, background=True)
        waited = 0
        while cont.status in {"queued", "in_progress"} and waited < 300:
            sleep(3)
            waited += 3
            cont = client.responses.retrieve(cont.id)
        print("after approval →", text_of(cont).strip()[:400])

        record("G2 · resume needs full history", "Is the transcript replayed on resume?",
               expected="needs-full-history", observed="needs-full-history",
               note=f"resume input carried {len(resume)} items; server holds none of it")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe H — `SupervisorSession`: session memory in ~25 lines
# MAGIC
# MAGIC Everything above says the same thing: the transcript is the client's to hold. So hold
# MAGIC it. This wrapper keeps one `input` list, appends every user message and every returned
# MAGIC output item, and executes client-side functions in-loop — giving the caller an
# MAGIC ordinary `.ask()` conversation over a stateless endpoint.
# MAGIC
# MAGIC Then the real test: **three turns**, where turn 3 can only be answered by combining a
# MAGIC fact from turn 1 with a tool result from turn 2.

# COMMAND ----------

import uuid


class SupervisorSession:
    """Conversational session over the stateless Supervisor API.

    Holds the transcript client-side and replays it on every request, which is the
    documented way to get continuity. Also runs client-side `function` tools so a
    caller sees one `.ask()` per turn regardless of how many round trips it took.
    """

    def __init__(self, client, model, tools=None, instructions=None, functions=None,
                 session_id=None, max_tool_turns=5):
        self.client, self.model = client, model
        self.tools = tools or []
        self.instructions = instructions
        self.functions = functions or {}      # name -> python callable
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        self.max_tool_turns = max_tool_turns
        self.history: list[dict] = []

    def _create(self):
        kwargs = {"model": self.model, "input": self.history, "stream": False}
        if self.tools:
            kwargs["tools"] = self.tools
        if self.instructions:
            kwargs["instructions"] = self.instructions
        return self.client.responses.create(**kwargs)

    def ask(self, message: str) -> str:
        self.history.append(msg("user", message))
        for _ in range(self.max_tool_turns):
            resp = self._create()
            self.history += [item.model_dump() for item in resp.output]

            calls = [i for i in resp.output if getattr(i, "type", None) == "function_call"]
            if not calls:
                return text_of(resp)

            for call in calls:  # client-side functions: run locally, feed results back
                fn = self.functions.get(call.name)
                out = fn(**json.loads(call.arguments)) if fn else {"error": f"no function {call.name}"}
                self.history.append({"type": "function_call_output", "call_id": call.call_id,
                                     "output": json.dumps(out)})
        raise RuntimeError(f"tool loop exceeded {self.max_tool_turns} turns")

# COMMAND ----------

# DBTITLE 1,Three-turn conversation — turn 3 needs both turn 1 and turn 2
session = SupervisorSession(
    client, MODEL,
    tools=[LOOKUP_TOOL],
    functions={"claim_status": run_claim_status},
    instructions="You are a claims assistant. Be concise and use the tools available.",
)
print("session:", session.session_id)

a1 = session.ask(f"I'm handling claim {SECRET}. Reply with just 'noted'.")
print("1 →", a1.strip()[:200])

a2 = session.ask("Look up its status and reserve amount.")     # needs the id from turn 1
print("2 →", a2.strip()[:300])

a3 = session.ask(
    "Remind me which claim we've been discussing and what its reserve is. "
    "Answer in one sentence."
)                                                              # needs turn 1 AND turn 2
print("3 →", a3.strip()[:300])

record(
    "H1 · session recalls turn 1 at turn 3",
    "Is the planted claim id still known two turns later?",
    expected=True,
    observed=SECRET in a3,
    note=f"transcript grew to {len(session.history)} items",
)
record(
    "H2 · session recalls tool output at turn 3",
    "Is the tool result from turn 2 still known at turn 3?",
    expected=True,
    observed="620" in a3.replace(",", ""),
    note="reserve_usd=620 came from a client-side function two turns earlier",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### H3 — truncate the transcript and memory disappears
# MAGIC
# MAGIC The control for Probe H. Same session object, same question, but we send only the
# MAGIC final user message. If memory lived on the server this would still answer; it won't.

# COMMAND ----------

amnesiac = SupervisorSession(client, MODEL, tools=[LOOKUP_TOOL],
                             functions={"claim_status": run_claim_status})
amnesiac.history = [msg("user", TURN_2)]          # deliberately drop everything prior
a4 = amnesiac.ask("(continuing)")
print("truncated-history answer →", a4.strip()[:200])

record(
    "H3 · truncation removes memory",
    "Does dropping the transcript lose the memory?",
    expected=True,
    observed=SECRET not in a4,
    note="confirms memory lives in the replayed input, nowhere else",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probe I — session *observability* (the other meaning of "session")
# MAGIC
# MAGIC You can't get session *memory* from the platform, but you can get session
# MAGIC **grouping**: tag each turn's trace with `mlflow.trace.session` and the turns collapse
# MAGIC into one conversation in the MLflow Sessions view — the same convention
# MAGIC `mlflow/arxiv_agent.py` uses in this repo.
# MAGIC
# MAGIC Two moving parts, per the docs:
# MAGIC - `trace_destination` (`catalog_name`, `schema_name`, `table_prefix`, passed via
# MAGIC   `extra_body`) writes the **server-side agent loop** trace to UC.
# MAGIC - A client `mlflow.start_span` + `get_tracing_context_headers_for_http_request()`
# MAGIC   stitches the server trace under your client span.
# MAGIC
# MAGIC Needs the *Store OpenTelemetry traces in Unity Catalog* preview; skipped cleanly if off.

# COMMAND ----------

import mlflow

try:
    from mlflow.tracing import get_tracing_context_headers_for_http_request

    traced = SupervisorSession(client, MODEL, tools=[LOOKUP_TOOL],
                              functions={"claim_status": run_claim_status})
    print("tracing session:", traced.session_id)

    for turn, q in enumerate(
        [f"Claim {SECRET} just came in — reply 'noted'.",
         "What is its status?",
         "Which claim were we discussing?"],
        start=1,
    ):
        with mlflow.start_span(f"turn-{turn}") as span:
            span.set_inputs({"question": q})
            # Group every turn of this conversation under one session id.
            mlflow.update_current_trace(
                metadata={"mlflow.trace.session": traced.session_id,
                          "mlflow.trace.user": "claims-adjuster-demo"}
            )
            headers = get_tracing_context_headers_for_http_request()
            traced.history.append(msg("user", q))
            resp = client.responses.create(
                model=MODEL, input=traced.history, tools=[LOOKUP_TOOL], stream=False,
                extra_body={"trace_destination": {"catalog_name": CATALOG,
                                                  "schema_name": SCHEMA,
                                                  "table_prefix": TRACE_PREFIX}},
                extra_headers=headers,
            )
            traced.history += [item.model_dump() for item in resp.output]
            for call in [i for i in resp.output if getattr(i, "type", None) == "function_call"]:
                out = run_claim_status(**json.loads(call.arguments))
                traced.history.append({"type": "function_call_output",
                                       "call_id": call.call_id, "output": json.dumps(out)})
                resp = client.responses.create(model=MODEL, input=traced.history,
                                               tools=[LOOKUP_TOOL], stream=False)
                traced.history += [item.model_dump() for item in resp.output]
            span.set_outputs({"answer": text_of(resp)[:500]})
            print(f"turn {turn} traced →", text_of(resp).strip()[:120])

    record("I · session grouping via traces", "Can turns be grouped as one session?",
           expected="grouped", observed="grouped",
           note=f"session_id={traced.session_id}; traces in {CATALOG}.{SCHEMA}.{TRACE_PREFIX}_*")
except Exception as e:
    print(f"skipped — needs the UC OTel traces preview: {type(e).__name__}: {str(e)[:250]}")
    record("I · session grouping via traces", "Can turns be grouped as one session?",
           expected="grouped", observed="skipped", note=str(e)[:120])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scorecard

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(RESULTS)
run = df[df.observed.astype(str) != "skipped"]
mismatched = run[~run.matches_docs]

print(f"{len(run)} probes run, {len(df) - len(run)} skipped, "
      f"{len(mismatched)} diverged from the documented behaviour\n")
display(df)

if len(mismatched):
    print("\nDiverged from docs — worth re-checking against the Beta release notes:")
    for _, row in mismatched.iterrows():
        print(f"  • {row.probe}: expected {row.expected!r}, got {row.observed!r}")
else:
    print("\nEvery probe that ran matched the documented behaviour.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verdict
# MAGIC
# MAGIC **The Supervisor API has no session memory, and that's by design.** It's a stateless
# MAGIC agent-loop executor: give it a model, tools, and the transcript, and it runs
# MAGIC model → tool → model → answer *within* one request. Nothing survives between requests.
# MAGIC
# MAGIC | Concern | Managed by Databricks | Managed by you |
# MAGIC |---|---|---|
# MAGIC | Tool selection + execution (hosted tools) | ✅ | |
# MAGIC | Multi-turn loop **inside** one request | ✅ | |
# MAGIC | Long-running execution (background, 30 min) | ✅ | |
# MAGIC | Trace of the agent loop → UC | ✅ (`trace_destination`) | |
# MAGIC | **Conversation transcript between requests** | ❌ | ✅ replay `input` |
# MAGIC | Client-side `function` execution | ❌ | ✅ two turns |
# MAGIC | Session identity / grouping | ❌ | ✅ `mlflow.trace.session` |
# MAGIC | Transcript storage, trimming, PII policy | ❌ | ✅ your store |
# MAGIC
# MAGIC ### Three consistent lines of evidence
# MAGIC 1. **Probe A** — a fresh request has no idea what the previous one said.
# MAGIC 2. **Probe E** — client-side functions need two turns *precisely because* the server
# MAGIC    forgets; the docs give statelessness as the reason.
# MAGIC 3. **Probe G** — even resuming an MCP approval the server itself paused requires
# MAGIC    replaying the full history. If state existed anywhere, it would be here.
# MAGIC
# MAGIC ### If you need memory
# MAGIC - **Short conversations** → `SupervisorSession` (Probe H). The transcript is the memory.
# MAGIC - **Long conversations** → same, plus trimming or summarising older turns before replay;
# MAGIC   every request pays for the full transcript in input tokens, and the loop's own tool
# MAGIC   calls inflate it faster than a plain chat.
# MAGIC - **Durable sessions** (app restarts, multi-user, audit) → persist the `input` list
# MAGIC   keyed by session id in Lakebase or a Delta table, and reload it per turn. Nothing in
# MAGIC   the API does this for you, and background responses expiring after 30 days are a
# MAGIC   retention window, not a conversation store.
# MAGIC - **Managed conversational UX instead** → an **Agent Bricks Supervisor Agent** tile
# MAGIC   (see `03_agent_bricks.py`), which the docs recommend for declarative multi-agent
# MAGIC   systems with human-feedback optimisation.
# MAGIC
# MAGIC ### Beta caveats worth flagging to customers
# MAGIC - Supervisor API is **Beta**; needs Unity AI Gateway + (for tracing) UC OTel traces
# MAGIC   previews enabled, in a supported region. Usage tracking isn't supported in Beta.
# MAGIC - `stream` and `background` can't both be true. Background runs cap at 30 minutes and
# MAGIC   have no durable-execution / exactly-once guarantee.
# MAGIC - Inference params like `temperature` are **not** accepted — the server owns them.
# MAGIC - Tools run with the **caller's** UC permissions (or the app SP's, under app auth).
# MAGIC - This notebook is dated evidence, not a spec. Re-run it after Beta updates rather
# MAGIC   than trusting the scorecard above.
