# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the reward function (no server / GPU needed): pytest tests/"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# import the pure reward fn without importing nemo_gym (guard the framework import)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_app", pathlib.Path(__file__).resolve().parents[1] / "app.py")


def _load_score():
    src = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text()
    ns = {}
    # exec only the pure-python bits we need (score_orchestration + helpers + consts)
    start = src.index("# --- reward weights")
    end = src.index("class BankingOrchestrationConfig")
    exec("import json\nfrom typing import Any, Dict, List\n" + src[start:end], ns)
    return ns["score_orchestration"]


score = _load_score()


def calls(*names):
    return [{"name": n, "arguments": {}} for n in names]


def test_clean_dispute_full_reward():
    meta = {"intent": "dispute_transaction", "required_agents": ["check_fraud_flags", "open_dispute"]}
    s = score(meta, calls("verify_identity", "check_fraud_flags", "open_dispute"))
    assert s["resolved"] == 1.0 and s["violations"] == 0.0
    assert s["reward"] > 0.9


def test_dispute_without_identity_is_penalized():
    meta = {"intent": "dispute_transaction", "required_agents": ["check_fraud_flags", "open_dispute"]}
    s = score(meta, calls("check_fraud_flags", "open_dispute"))  # no verify_identity
    assert s["violations"] >= 1.0
    assert s["reward"] < score(meta, calls("verify_identity", "check_fraud_flags", "open_dispute"))["reward"]


def test_lost_card_must_block():
    meta = {"intent": "lost_card", "required_agents": ["block_card"]}
    assert score(meta, calls("verify_identity"))["violations"] >= 1.0
    assert score(meta, calls("verify_identity", "block_card"))["violations"] == 0.0


def test_redundant_calls_reduce_efficiency():
    meta = {"intent": "balance_inquiry", "required_agents": ["get_balance"]}
    lean = score(meta, calls("get_balance"))
    noisy = score(meta, calls("get_balance", "check_fraud_flags", "transfer_funds"))
    assert lean["efficiency"] > noisy["efficiency"]


if __name__ == "__main__":
    for fn in [test_clean_dispute_full_reward, test_dispute_without_identity_is_penalized,
               test_lost_card_must_block, test_redundant_calls_reduce_efficiency]:
        fn(); print("ok:", fn.__name__)
    print("ALL PASS")
