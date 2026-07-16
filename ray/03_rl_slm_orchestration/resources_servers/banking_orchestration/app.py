# SPDX-License-Identifier: Apache-2.0
"""
Banking Orchestration Resources Server (NeMo Gym).

FinServ agent-orchestration environment. The policy model (Qwen3-4B) acts as an
**orchestrator**: given a customer servicing request, it must call the right
*specialist agents* (exposed as tools) in a *policy-compliant order* and produce a
resolution. RL trains the SLM to orchestrate well.

Specialist agents (tools):
  - verify_identity(customer_id)        -> KYC / identity agent
  - check_fraud_flags(account_id)       -> fraud agent
  - get_balance(account_id)             -> accounts agent
  - open_dispute(txn_id, reason)        -> disputes agent
  - block_card(card_id)                 -> card-services agent
  - transfer_funds(from, to, amount)    -> payments agent
  - escalate_to_human(reason)           -> escalation agent

Compliance policy enforced by the verifier (this is the "specific orchestration"
the reward shapes):
  1. verify_identity MUST precede any account-modifying action
     (open_dispute, block_card, transfer_funds).
  2. check_fraud_flags MUST precede open_dispute.
  3. A lost/stolen card MUST result in block_card.
  4. Resolve with the fewest correct steps; penalize wrong/redundant agent calls.

NOTE: base-class method signatures follow NeMo Gym `SimpleResourcesServer`. Confirm
against the pinned nemo_gym version (`uv pip show nemo_gym`) — the reward math below
is framework-independent, only the request/response plumbing is version-sensitive.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

# --- reward weights ---------------------------------------------------------
R_RESOLVED = 0.5          # correct terminal resolution for the intent
R_POLICY = 0.4            # full compliance-policy adherence
R_EFFICIENCY = 0.1        # minimal / no redundant agent calls
PENALTY_VIOLATION = 0.5   # subtracted per hard policy violation

ACCOUNT_MODIFYING = {"open_dispute", "block_card", "transfer_funds"}


def _tool_calls(rollout: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ordered function/tool calls from a NeMo Gym rollout's output items."""
    calls: List[Dict[str, Any]] = []
    for item in rollout.get("output", []) or []:
        if item.get("type") in ("function_call", "tool_call"):
            name = item.get("name") or item.get("function", {}).get("name")
            raw = item.get("arguments") or item.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": name, "arguments": args})
    return calls


def score_orchestration(task_meta: Dict[str, Any], calls: List[Dict[str, Any]]) -> Dict[str, float]:
    """Pure reward function — unit-testable without the server (see tests/)."""
    names = [c["name"] for c in calls]
    intent = task_meta["intent"]
    required = set(task_meta.get("required_agents", []))
    violations = 0

    # Policy 1: identity before any account-modifying action
    if any(n in ACCOUNT_MODIFYING for n in names):
        first_mod = min(names.index(n) for n in names if n in ACCOUNT_MODIFYING)
        if "verify_identity" not in names or names.index("verify_identity") > first_mod:
            violations += 1
    # Policy 2: fraud check before opening a dispute
    if "open_dispute" in names:
        if "check_fraud_flags" not in names or names.index("check_fraud_flags") > names.index("open_dispute"):
            violations += 1
    # Policy 3: lost/stolen card must block the card
    if intent == "lost_card" and "block_card" not in names:
        violations += 1

    resolved = required.issubset(set(names))
    # efficiency: 1.0 if no calls outside the required set, decaying with extras
    extras = [n for n in names if n not in required and n != "verify_identity"]
    efficiency = max(0.0, 1.0 - 0.25 * len(extras))

    reward = (
        R_RESOLVED * (1.0 if resolved else 0.0)
        + R_POLICY * (1.0 if violations == 0 else 0.0)
        + R_EFFICIENCY * efficiency
        - PENALTY_VIOLATION * violations
    )
    return {
        "reward": max(0.0, min(1.0, reward)),
        "resolved": float(resolved),
        "violations": float(violations),
        "efficiency": efficiency,
    }


class BankingOrchestrationConfig(BaseResourcesServerConfig):
    judge_disabled: bool = Field(default=True, description="Rule-based verifier; no LLM judge needed.")


class BankingOrchestrationServer(SimpleResourcesServer):
    """Rule-based verifier for FinServ agent orchestration."""

    config: BankingOrchestrationConfig

    async def verify(self, request: BaseVerifyRequest) -> BaseVerifyResponse:
        # `request.rollout` is the agent's Responses-API rollout; task metadata is
        # carried on the task under `extra_info` / `metadata` (set by the generator).
        rollout = request.rollout if isinstance(request.rollout, dict) else request.rollout.model_dump()
        task_meta = (request.extra_info or {}).get("orchestration") or {}
        calls = _tool_calls(rollout)
        scored = score_orchestration(task_meta, calls)
        return BaseVerifyResponse(reward=scored["reward"], metrics=scored)


app = BankingOrchestrationServer.build_app(BankingOrchestrationConfig)
