# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# ///

# MAGIC %md
# MAGIC # 02 · GRPO RL training on AI Runtime + Ray
# MAGIC
# MAGIC Trains **Qwen3-4B** to orchestrate the banking specialist agents, optimizing the
# MAGIC compliance-policy reward from `resources_servers/banking_orchestration/app.py`.
# MAGIC **GRPO** (group-relative, no value model) via **NeMo RL**; **Ray** fans rollouts
# MAGIC across many concurrent NeMo Gym envs; runs on **AI Runtime serverless GPU**
# MAGIC (single 8xH100 node ideal; 1xA10 works as a LoRA smoke test).

# COMMAND ----------

# MAGIC %pip install --quiet "nemo-gym @ git+https://github.com/NVIDIA-NeMo/Gym.git" "nemo-rl" mlflow
# MAGIC %restart_python

# COMMAND ----------

import os, ray, mlflow
ray.init(ignore_reinit_error=True)
mlflow.set_experiment("/Users/scott.mckean@databricks.com/rl_slm_orchestration")
print("Ray:", ray.cluster_resources())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch GRPO
# MAGIC The trainer (a) spins vLLM for on-policy generation, (b) drives the NeMo Gym
# MAGIC `banking_orchestration_agent` for rollouts (reward = policy compliance +
# MAGIC resolution + efficiency), (c) computes group-relative advantages, (d) updates
# MAGIC the LoRA policy. Config: `train/grpo_qwen3.yaml`.

# COMMAND ----------

import subprocess
with mlflow.start_run(run_name="grpo_qwen3_4b_orchestration"):
    mlflow.log_artifact("../train/grpo_qwen3.yaml")
    # NeMo RL entrypoint; --gym-config points it at this repo's environment/agent.
    subprocess.run([
        "python", "-m", "nemo_rl.train",
        "--config", "train/grpo_qwen3.yaml",
        "--gym-config", "config.yaml",
        "--agent", "banking_orchestration_agent",
    ], cwd="..", check=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## After training
# MAGIC - Register the tuned adapter to Unity Catalog and serve it (Model Serving) as the
# MAGIC   production orchestrator; the same NeMo Gym eval (notebook 01) gives a
# MAGIC   trained-vs-baseline reward delta on held-out tasks.
# MAGIC - Scale up: set `cluster.num_gpus: 8` + `gpu_type: H100` and raise
# MAGIC   `rollout_workers` — NeMo Gym scales to thousands of concurrent envs.
