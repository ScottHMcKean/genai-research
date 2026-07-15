# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# ///

# MAGIC %md
# MAGIC # 01 · Serve Qwen3-4B on AI Runtime (vLLM) + init Ray
# MAGIC
# MAGIC Stands up the **policy model** NeMo Gym drives: Qwen3-4B behind an
# MAGIC OpenAI-compatible **vLLM** server, on **AI Runtime serverless GPU**, with **Ray**
# MAGIC for the rollout fan-out. Pinned to `databricks_ai_v5` (torch/CUDA/ray preinstalled);
# MAGIC submit with a GPU accelerator (see repo README — `compute.hardware_accelerator`).

# COMMAND ----------

# MAGIC %pip install --quiet vllm "nemo-gym @ git+https://github.com/NVIDIA-NeMo/Gym.git"
# MAGIC %restart_python

# COMMAND ----------

import torch, ray
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
# Ray is preinstalled on AI Runtime; init the local cluster used for rollout workers.
ray.init(ignore_reinit_error=True)
print("Ray resources:", ray.cluster_resources())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch the vLLM OpenAI server (Qwen3-4B)
# MAGIC On 1xA10 use `Qwen/Qwen3-1.7B`; on 1xH100 `Qwen/Qwen3-4B` fits comfortably.

# COMMAND ----------

import subprocess, sys, time, requests

MODEL = "Qwen/Qwen3-4B"           # ~3B-class Qwen3; swap to Qwen3-1.7B for A10
PORT = 8000
proc = subprocess.Popen([
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL, "--port", str(PORT),
    "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
])

base_url = f"http://localhost:{PORT}/v1"
for _ in range(60):                # wait for weights to load
    try:
        if requests.get(f"http://localhost:{PORT}/health", timeout=2).ok:
            break
    except Exception:
        time.sleep(10)
print("vLLM ready at", base_url, "-> set policy_base_url in ../config.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test — one orchestration rollout via NeMo Gym
# MAGIC Runs the untrained baseline against a few tasks so you can see the reward the
# MAGIC verifier assigns (this is the signal GRPO will optimize in notebook 02).

# COMMAND ----------

import subprocess
subprocess.run([
    "gym", "eval", "run", "--no-serve",
    "--agent", "banking_orchestration_agent",
    "--input", "resources_servers/banking_orchestration/data/example.jsonl",
    "--output", "results/baseline_rollouts.jsonl",
    "--limit", "8", "--num-repeats", "1",
], cwd="..", check=False)
# Inspect results/baseline_rollouts.jsonl -> each row carries the reward + metrics
# (resolved / violations / efficiency) from app.py's verifier.
