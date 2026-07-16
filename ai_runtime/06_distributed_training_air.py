# Databricks notebook source

# MAGIC %md
# MAGIC # 06 · AI Runtime — Distributed Training on Serverless GPU
# MAGIC
# MAGIC Demonstrates the **AI Runtime** `serverless_gpu` API. This notebook is pinned (header
# MAGIC above) to the **`databricks_ai_v5`** GPU base environment, so the `serverless_gpu`
# MAGIC library, CUDA, and PyTorch are preinstalled and the `@distributed` call provisions an
# MAGIC **A10** on demand — no cluster to manage.
# MAGIC
# MAGIC Reference: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training
# MAGIC
# MAGIC - `gpus`: number of GPUs (single node; `gpus=8, gpu_type='H100'` for 8xH100).
# MAGIC - `gpu_type`: `'A10'` or `'H100'` (auto-detected if omitted).
# MAGIC - Each `.distributed()` call auto-creates an **MLflow run** (link printed in output).
# MAGIC - Requires **GPU environment 4+** (v5 here).

# COMMAND ----------

from serverless_gpu import distributed
import torch, torch.nn as nn
print("driver torch:", torch.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Single-GPU (A10) training function
# MAGIC A tiny classifier on synthetic data — self-contained, finishes in seconds — to prove
# MAGIC the GPU is real and the API works end-to-end.

# COMMAND ----------

@distributed(gpus=1, gpu_type='A10')
def train():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    print("remote device:", dev, "|", name)
    torch.manual_seed(0)
    x = torch.randn(4096, 32, device=dev)
    y = (x.sum(1, keepdim=True) > 0).float()
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    for epoch in range(300):
        opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
    acc = ((model(x) > 0).float() == y).float().mean().item()
    print(f"final loss={loss.item():.4f}  train_acc={acc:.3f}")
    return {"device": name, "loss": loss.item(), "acc": acc}

# COMMAND ----------

result = train.distributed()
print("returned from GPU worker:", result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scaling up
# MAGIC Swap to `@distributed(gpus=8, gpu_type='H100')` for a single 8xH100 node; the same
# MAGIC function body runs data-parallel across the 8 GPUs. See the doc link above.
