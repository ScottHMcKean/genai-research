# SPDX-License-Identifier: Apache-2.0
"""
Generate synthetic FinServ servicing tasks in NeMo Gym Responses-API format.

Each line = one task the orchestrator must solve:
  - developer message: the orchestrator system prompt + compliance policy
  - user message: a customer servicing request
  - tools: the specialist-agent function schemas
  - extra_info.orchestration: gold metadata the verifier uses (intent + required_agents)

Usage:
  python generate_tasks.py --n 512 --out data/train.jsonl
  python generate_tasks.py --n 64  --out data/example.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

SYSTEM = (
    "You are a retail-bank servicing ORCHESTRATOR. Resolve the customer's request by "
    "calling specialist-agent tools. Compliance policy you MUST follow:\n"
    "1. Call verify_identity before ANY account-modifying action "
    "(open_dispute, block_card, transfer_funds).\n"
    "2. Call check_fraud_flags before open_dispute.\n"
    "3. A lost/stolen card MUST result in block_card.\n"
    "Use the fewest correct steps. Do not call agents you do not need."
)

TOOLS = [
    ("verify_identity", {"customer_id": "string"}, "KYC/identity agent: verify the caller."),
    ("check_fraud_flags", {"account_id": "string"}, "Fraud agent: return open fraud flags."),
    ("get_balance", {"account_id": "string"}, "Accounts agent: return current balance."),
    ("open_dispute", {"txn_id": "string", "reason": "string"}, "Disputes agent: open a dispute."),
    ("block_card", {"card_id": "string"}, "Card-services agent: block a card."),
    ("transfer_funds", {"from_account": "string", "to_account": "string", "amount": "number"}, "Payments agent: move funds."),
    ("escalate_to_human", {"reason": "string"}, "Escalation agent: hand off to a human."),
]

# intent -> (customer utterance templates, required specialist agents for resolution)
INTENTS = {
    "balance_inquiry": (["What's my checking balance?", "How much do I have available?"], ["get_balance"]),
    "dispute_transaction": (["I want to dispute a ${amt} charge from {merch}.",
                              "There's a charge I don't recognize for ${amt}."],
                             ["check_fraud_flags", "open_dispute"]),
    "lost_card": (["I lost my debit card.", "My card was stolen last night."], ["block_card"]),
    "transfer_funds": (["Move ${amt} from checking to savings.", "Transfer ${amt} to my savings."],
                       ["transfer_funds"]),
    "fraud_alert": (["I got a fraud alert text — is my account okay?"], ["check_fraud_flags", "escalate_to_human"]),
}

MERCHANTS = ["QuickMart", "AirFly", "StreamCo", "GasNGo", "MegaStore"]


def tool_schema():
    out = []
    for name, props, desc in TOOLS:
        out.append({
            "type": "function", "name": name, "description": desc, "strict": True,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v, "description": ""} for k, v in props.items()},
                "required": list(props.keys()), "additionalProperties": False,
            },
        })
    return out


def make_task(rng: random.Random) -> dict:
    intent = rng.choice(list(INTENTS.keys()))
    templates, required = INTENTS[intent]
    utter = rng.choice(templates).format(amt=rng.choice([12, 40, 250, 999]), merch=rng.choice(MERCHANTS))
    return {
        "responses_create_params": {
            "input": [
                {"role": "developer", "content": SYSTEM},
                {"role": "user", "content": utter},
            ],
            "tools": tool_schema(),
        },
        "extra_info": {"orchestration": {"intent": intent, "required_agents": required}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w") as f:
        for _ in range(args.n):
            f.write(json.dumps(make_task(rng)) + "\n")
    print(f"wrote {args.n} tasks -> {args.out}")


if __name__ == "__main__":
    main()
