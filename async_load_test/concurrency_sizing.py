# Databricks notebook source
# MAGIC %md
# MAGIC # How fast can a small serverless CPU *fire* async requests?
# MAGIC
# MAGIC The compute needed to **serve** 200 concurrent requests lives on the *endpoint*. This notebook profiles
# MAGIC the other half: how fast the **client** — a small serverless CPU — can *fire and hold* requests with
# MAGIC `asyncio` + a `Semaphore`. We simulate each endpoint call with `asyncio.sleep(100–300 ms)`, so the only
# MAGIC thing being measured is the client's dispatch cost — no network, no endpoint.
# MAGIC
# MAGIC Two numbers come out:
# MAGIC 1. **Max fire rate** — requests/s the event loop can push through the async machinery (the ceiling).
# MAGIC 2. **Fire rate needed** for your target — `concurrency / mean_latency` ≈ `200 / 0.2s` ≈ **1000 req/s**.
# MAGIC
# MAGIC If the ceiling sits comfortably above what's needed, a small CPU has the capacity to fire 200-way
# MAGIC concurrency — the firing is cheap, the *serving* is where the compute goes.
# MAGIC
# MAGIC > Runs on **serverless CPU** (shm-skunkworks). Simulate-only — no endpoint required.

# COMMAND ----------

# MAGIC %pip install nest_asyncio==1.6.* -q
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("target_concurrency", "200", "Target concurrency (semaphore limit)")
dbutils.widgets.text("total_requests", "20000", "Requests to fire per measurement")
dbutils.widgets.text("latency_min_ms", "100", "Simulated endpoint latency: min (ms)")
dbutils.widgets.text("latency_max_ms", "300", "Simulated endpoint latency: max (ms)")

TARGET = int(dbutils.widgets.get("target_concurrency"))
TOTAL = int(dbutils.widgets.get("total_requests"))
LAT_MIN = int(dbutils.widgets.get("latency_min_ms")) / 1000
LAT_MAX = int(dbutils.widgets.get("latency_max_ms")) / 1000

# COMMAND ----------

import asyncio, os, random, time
import numpy as np
import nest_asyncio

# Databricks cells run inside a live event loop; nest_asyncio lets us re-enter it so asyncio.run() works.
nest_asyncio.apply()

N_CORES = os.cpu_count()
print(f"driver cores available: {N_CORES}")
print(f"target_concurrency={TARGET}  total_requests={TOTAL}  sim_latency={int(LAT_MIN*1000)}-{int(LAT_MAX*1000)}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Max fire rate — the client ceiling
# MAGIC Fire every request through the real async path (task + `Semaphore` acquire/release) but with a **no-op**
# MAGIC "endpoint" (`sleep(0)`). Wall time is pure client dispatch cost, so `requests / wall` is the ceiling on how
# MAGIC fast this CPU can push requests out the door.

# COMMAND ----------

async def fire(total, concurrency, work):
    """Fire `total` requests capped at `concurrency`. `work` is the awaited per-request 'endpoint' call."""
    sem = asyncio.Semaphore(concurrency)

    async def one():
        async with sem:            # the exact concurrency cap you use in testing
            return await work()

    cpu0, wall0 = time.process_time(), time.perf_counter()
    await asyncio.gather(*(one() for _ in range(total)))
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return wall, cpu


async def noop():
    await asyncio.sleep(0)

wall, cpu = asyncio.run(fire(TOTAL, TARGET, noop))
max_fire_rps = TOTAL / wall
cpu_us_per_req = cpu / TOTAL * 1e6
print(f"fired {TOTAL} requests in {wall:.2f}s")
print(f"MAX FIRE RATE  = {max_fire_rps:,.0f} req/s")
print(f"CPU per fire   = {cpu_us_per_req:,.1f} µs  (event-loop dispatch cost per request)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Hold 200 concurrent with realistic latency
# MAGIC Same driver, but each "request" now waits 100–300 ms like your endpoints. We confirm we actually reach
# MAGIC 200 in-flight and that the CPU stays nearly idle while all that waiting happens.

# COMMAND ----------

inflight = 0
peak = 0

async def sim_request():
    global inflight, peak
    inflight += 1
    peak = max(peak, inflight)
    try:
        await asyncio.sleep(random.uniform(LAT_MIN, LAT_MAX))  # pure I/O wait — no CPU
    finally:
        inflight -= 1

peak = 0
wall, cpu = asyncio.run(fire(TOTAL, TARGET, sim_request))
sustained_rps = TOTAL / wall
cores_used = cpu / wall
print(f"peak in-flight reached = {peak}  (asked for {TARGET})")
print(f"sustained throughput   = {sustained_rps:,.0f} req/s")
print(f"event loop burned      = {cores_used:.3f} of 1 core while holding {peak} concurrent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verdict

# COMMAND ----------

mean_latency_s = (LAT_MIN + LAT_MAX) / 2
needed_fire_rps = TARGET / mean_latency_s          # fire rate required to keep TARGET in flight
headroom = max_fire_rps / needed_fire_rps

print(f"To hold {TARGET}-way concurrency at ~{mean_latency_s*1000:.0f} ms latency, fire ~{needed_fire_rps:,.0f} req/s.")
print(f"This CPU can fire up to ~{max_fire_rps:,.0f} req/s.")
print(f"Headroom = {headroom:,.0f}x")
print()
if headroom >= 1:
    print(f"=> YES — a small serverless CPU has plenty of capacity to FIRE {TARGET}-way concurrency.")
    print(f"   Firing is cheap ({cpu_us_per_req:,.0f} µs/req); the compute you need is on the ENDPOINT,")
    print(f"   which has to actually serve {needed_fire_rps:,.0f} req/s.")
else:
    print(f"=> The client is the bottleneck: it can't fire fast enough for {TARGET}-way concurrency.")
    print(f"   Split the load across processes (one asyncio loop = one core).")
