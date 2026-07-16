# RL for Agent Orchestration — FinServ SLM (Qwen3-4B) on AI Runtime + Ray

Reinforcement-learning example: train a **small language model (Qwen3-4B)** to be a
**retail-bank servicing orchestrator** — it decides *which specialist agents to call,
in what order*, under a compliance policy — using **[NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)**
for the environment/reward and **GRPO** (via NeMo RL) for training, all on
**Databricks AI Runtime (serverless GPU) with Ray**.

## Why this matters for a bank
Most FinServ AI at scale is **narrow, high-volume, policy-bound** orchestration —
route the request, verify identity, check fraud, resolve. A **fine-tuned SLM
orchestrator** matches or beats a frontier LLM on these tasks at a fraction of the
cost/latency, runs on-platform under UC governance, and — crucially — **RL lets you
bake the compliance policy into the reward** instead of hoping a prompt holds.

## The task (specific agent orchestration)
The SLM orchestrates seven specialist agents (exposed as tools): `verify_identity`
(KYC), `check_fraud_flags` (fraud), `get_balance` (accounts), `open_dispute`
(disputes), `block_card` (card services), `transfer_funds` (payments),
`escalate_to_human` (escalation).

**Compliance policy shaped by the reward** (`resources_servers/banking_orchestration/app.py`):
1. `verify_identity` **before** any account-modifying action.
2. `check_fraud_flags` **before** `open_dispute`.
3. A lost/stolen card **must** `block_card`.
4. Resolve in the fewest correct steps; wrong/redundant agent calls are penalized.

Reward = `0.5·resolved + 0.4·policy_compliant + 0.1·efficiency − 0.5·violations`,
clamped to `[0,1]`. Rule-based (fast, deterministic, auditable) — no LLM judge.

## Architecture
```
Databricks AI Runtime (serverless GPU, env databricks_ai_v5)
├── vLLM  ── serves Qwen3-4B as an OpenAI-compatible policy endpoint   (nb 01)
├── Ray   ── fans rollouts across many concurrent NeMo Gym envs
└── NeMo Gym
    ├── resources_servers/banking_orchestration  (env + compliance reward)
    └── responses_api_agents/simple_agent        (multi-step tool-calling harness)
        ↓ rollouts + rewards
    NeMo RL · GRPO ── group-relative advantages → LoRA update on Qwen3-4B  (nb 02)
        ↓
    MLflow + Unity Catalog ── track, register, serve the tuned orchestrator
```

## Layout
| Path | What |
|------|------|
| `resources_servers/banking_orchestration/app.py` | NeMo Gym env server + compliance-policy verifier (reward) |
| `resources_servers/banking_orchestration/data/generate_tasks.py` | Synthetic FinServ servicing tasks (Responses-API format) |
| `resources_servers/banking_orchestration/configs/banking_orchestration.yaml` | Env ↔ agent ↔ policy-model wiring |
| `resources_servers/banking_orchestration/tests/test_app.py` | Reward-function unit tests (no GPU) |
| `config.yaml` | Top-level: points the policy model at the vLLM endpoint |
| `train/grpo_qwen3.yaml` | GRPO / NeMo RL trainer config (LoRA, Ray, 8×H100 or 1×A10) |
| `notebooks/01_air_ray_serve_qwen_vllm.py` | Serve Qwen3-4B (vLLM) + init Ray + baseline rollout |
| `notebooks/02_grpo_train_air_ray.py` | GRPO training on AI Runtime + Ray |
| `notebooks/03_smoke_test_air.py` | **Verified** end-to-end smoke test on A10 (transformers + reward + MLflow) |

## Run it
1. **Env**: run the notebooks on **AI Runtime** pinned to `databricks_ai_v5` with a GPU
   accelerator. Headless job: `environments[].spec.environment_version: "5"` **plus**
   task `compute: {hardware_accelerator: "GPU_1xA10"}` (or `GPU_8xH100`).
2. **Tests** (local, no GPU): `python resources_servers/banking_orchestration/tests/test_app.py`
3. **Smoke test (start here)**: run `notebooks/03` on an A10 → loads Qwen3-1.7B, runs the
   tasks through tool-calling, scores with the verifier, logs the baseline to MLflow.
   Confirms the GPU + policy-model + tool-calling + reward loop works end-to-end.
4. **Data**: `python resources_servers/banking_orchestration/data/generate_tasks.py --n 512 --out .../data/train.jsonl`
5. **Serve + baseline (scale)**: `notebooks/01` → vLLM serves Qwen3-4B, Ray up, NeMo Gym rollouts.
6. **Train**: `notebooks/02` → GRPO; watch `mean_reward` / `violations` trend in the same MLflow experiment.
7. **Ship**: register the LoRA adapter to UC, serve via Model Serving, re-run the eval for a trained-vs-baseline delta.

### Verified baseline
`notebooks/03` ran green on an **A10G** (untrained `Qwen/Qwen3-1.7B`) → MLflow experiment
`/Users/<you>/rl_slm_orchestration`: **`mean_reward = 0.85`**, `mean_violations = 0`,
`resolved_rate = 0.75`. The model already respects the compliance policy but fails to
resolve `transfer_funds` — exactly the resolution/efficiency gap **GRPO (nb 02) closes**.

## Model note
"Qwen3 ~3B" → this uses **`Qwen/Qwen3-4B`** (closest ~3B-class Qwen3). For a 1×A10
smoke test use **`Qwen/Qwen3-1.7B`**; for a full 8×H100 run, Qwen3-4B (or larger) with
`peft.enabled: false` for full fine-tuning.

## Status / caveats
- Reward logic is **tested and passing**. The `app.py` framework plumbing (base-class
  method signatures) targets NeMo Gym `SimpleResourcesServer` — **confirm against the
  pinned `nemo_gym` version** after `pip install`, as the API is pre-1.0.
- vLLM serving + NeMo RL entrypoints in the notebooks require the GPU env; they're
  written to run there, not validated on CPU.
- Reference: NeMo Gym `example_environments/example_single_tool_call` and
  `benchmarks/tau2/.../banking_alltools` (the tau2 banking tool-agent) informed this design.
