# Ray on Databricks — from basics to reinforcement learning

One narrative, three stops. Start where Ray starts — a plain Spark cluster — and walk all
the way to reinforcement-learning a small language model on AI Runtime. Each stop stands on
its own, but they read best in order.

| # | Notebook / folder | What it shows | Compute |
|---|-------------------|---------------|---------|
| 01 | [`01_ray_basics_classic_cluster.ipynb`](01_ray_basics_classic_cluster.ipynb) | **Basics.** Spin up a Ray cluster *on top of* a classic Spark cluster with `ray.util.spark.setup_ray_cluster`, fan a workload out across workers, and watch how parallelism and cluster memory interact. The "hello world" of Ray on Databricks. | Classic cluster |
| 02 | [`02_ray_external_model_inference.ipynb`](02_ray_external_model_inference.ipynb) | **Inference.** Task-based batch inference: use Ray to fan many rows out against an **external / OpenAI-compatible model endpoint**, the pattern for high-volume LLM inference that a single driver can't keep up with. | Classic or serverless |
| 03 | [`03_rl_slm_orchestration/`](03_rl_slm_orchestration/README.md) | **Reinforcement learning.** GRPO-train a **Qwen3 SLM** to be a compliant retail-bank agent orchestrator, using **NVIDIA NeMo Gym** for the environment/reward and **Ray** to fan rollouts across concurrent envs — all on **AI Runtime (serverless GPU)**. | Serverless GPU (AI Runtime) |

## The through-line

- **01 → why Ray at all.** Spark gives you data parallelism; Ray gives you task
  parallelism and arbitrary Python. On a classic cluster you can borrow the Spark workers
  as a Ray cluster and get both without leaving Databricks.
- **02 → the first real payoff.** Batch LLM inference is embarrassingly parallel but
  I/O-bound on one driver. Ray tasks turn a serial loop into a fan-out against a model
  endpoint — the same shape whether the model is external or served on-platform.
- **03 → the frontier.** Once you can fan rollouts out cheaply, you can *train* with them.
  RL (GRPO) rewards the model for calling the right tools in the right order under a
  compliance policy — Ray scales the rollouts, AI Runtime provides the GPUs, Unity Catalog
  + MLflow govern the result.

## Positioning (talk track)

> *Use Spark for the data plane, Ray for the compute plane, Databricks for the governed
> platform around both.*

Ray is **complementary to Spark, not a replacement** — Spark = data parallelism, Ray =
task parallelism + complex Python (simulation, HPO, distributed training, RL, batch
inference with heavy per-item logic). The differentiator is *Ray + the Lakehouse*: Unity
Catalog governance, MLflow, Jobs, and Spark ETL + Ray compute in one platform. For
GPU-backed Ray, **AI Runtime / Serverless GPU** is the preferred execution path.

For the fuller AI-Runtime story (Ray fan-out + Ray Data, LoRA fine-tuning on serverless
GPU, `serverless_gpu` distributed training) see the sibling [`ai_runtime/`](../ai_runtime/README.md)
folder and its [`REFERENCES.md`](../ai_runtime/REFERENCES.md) (public blogs/docs, FSI
use-cases, and the AI Runtime compliance caveat for regulated workloads).
