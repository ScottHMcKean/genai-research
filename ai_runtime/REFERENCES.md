# AI Runtime & Ray — reference material

Public references for the "AI Runtime, incl. Ray" material.

> **Naming:** **AI Runtime (AIR)** = **Serverless GPU Compute (SGC)** — serverless
> on-demand NVIDIA GPUs (A10/H100) integrated with Unity Catalog, Notebooks, Jobs, MLflow.
> **Ray** = the open-source distributed-compute framework. AIR is the preferred GPU
> execution path for Ray workloads that fit its current envelope.

## Public blogs & docs

- [Announcing GA of Ray on Databricks (blog)](https://www.databricks.com/blog/announcing-general-availability-ray-databricks)
- [Introducing AI Runtime: serverless NVIDIA GPUs for training & fine-tuning (blog)](https://www.databricks.com/blog/introducing-ai-runtime-scalable-serverless-nvidia-gpus-databricks-training-and-finetuning)
- [Announcing Ray support on Databricks & Spark clusters (blog)](https://www.databricks.com/blog/2023/02/28/announcing-ray-support-databricks-and-apache-spark-clusters.html)
- [What is Ray on Databricks? (docs)](https://docs.databricks.com/aws/en/machine-learning/ray)
- [AI Runtime (docs)](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)
- [Distributed training (docs)](https://docs.databricks.com/aws/en/machine-learning/train-model/distributed-training/)

## Talk track

**One-liner:** *Use Spark for the data plane, Ray for the compute plane, Databricks for
the governed platform around both.*

- **Ray is complementary to Spark, not a replacement.** Spark = data parallelism; Ray =
  task parallelism + complex Python (simulation, optimization, hyperparameter tuning,
  distributed training, RL, batch inference with heavy per-item logic).
- **The differentiator is "Ray + the Lakehouse"** — Unity Catalog governance, MLflow, Jobs,
  and Spark ETL + Ray compute in one governed platform.
- **For GPU-backed Ray, AI Runtime / SGC is the preferred path** — faster GPU access,
  pay-for-use, prebuilt DL packages (PyTorch/Transformers/Ray), simple orchestration.

**Insurance / FSI use-cases:** fraud & claims-anomaly models · claims severity triage &
adjuster-notes models (← this demo) · catastrophe / pricing / reserving / underwriting
simulation · fine-tuned assistants for adjusters, underwriters, brokers, advisors · AML
sequence/graph models · credit-risk & portfolio optimization.

**When NOT to use Ray:** ETL-heavy pipelines (use Spark) · online serving (use Model
Serving, not Ray Serve) · when the only need is the cheapest raw GPU · very large-scale
pretraining beyond AIR's sweet spot.

Public customer case studies (Ray on Databricks) are covered in the [Ray GA
blog](https://www.databricks.com/blog/announcing-general-availability-ray-databricks) and
Data + AI Summit talks — good sources for outcome/proof-point slides.

## ⚠️ Compliance caveat for regulated FSI workloads

Confirm current availability before committing on a slide: AI Runtime / SGC availability
can be limited by region and by compliance-security-profile (e.g. HIPAA/PCI) workspaces.
For regulated-data workloads in restricted environments, **Ray on classic Databricks
clusters** may be the near-term path; use AIR where the workload fits the supported
envelope. Check the current [AI Runtime docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)
for the latest supported regions and compliance status.

*(This demo runs Ray on serverless and does the LoRA fine-tune on serverless CPU when no
GPU is available; on an AI-Runtime-enabled workspace the same fine-tune runs on serverless GPU.)*
