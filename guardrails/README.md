# Don't Trust Your Guardrail — Test It

*3 min read · Notebook: [`guardrail_evaluation.ipynb`](guardrail_evaluation.ipynb)*

Every team shipping an LLM app adds a guardrail against prompt injection. Almost no one measures whether it works. "We added a safety filter" is a vibe, not a control. Here's how to turn it into a number — on Databricks, across datasets, two ways.

## Two places a guardrail lives

**Online — on the endpoint.** Mosaic AI Gateway runs safety, PII (block or mask), and keyword filters on every request and response. Fast, centralized, no app changes.

**Offline — a prompt.** The flexible Databricks pattern: *any model + a prompt* becomes an LLM-as-judge that classifies inputs as BLOCK or ALLOW. Great for nuanced injection *techniques* the endpoint filter doesn't model.

You want both. They catch different things, and layering them mirrors the gateway's own input/output guardrail split.

## The metric that actually matters

Everyone reports recall ("we catch 95% of attacks!"). The number that decides whether your guardrail survives production is the **false-positive rate** — how often it blocks *benign* traffic. A guardrail that flags "help me **decimate** my debt" gets switched off within a week.

So the eval set needs three buckets, and the third is the one people skip:
1. **Attacks** (expect BLOCK)
2. **Benign** prompts (expect ALLOW)
3. **Hard negatives** — benign text that *looks* like an attack

The notebook assembles all three from a bundled seed set plus reputable public data — [JailbreakBench](https://arxiv.org/abs/2404.01318), [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset), [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) — with graceful fallback so it runs even with no internet.

## What the notebook does

1. Builds and (optionally) governs a labeled eval set in Unity Catalog.
2. Runs the **offline judge** over every row and scores it with **MLflow 3 GenAI evaluation** — precision, recall, FPR, plus a per-technique breakdown of *where* it fails.
3. Replays the same data through the **online endpoint guardrail** and scores it identically.
4. Compares the two side by side.
5. **Aligns the judge with DSPy + GEPA** — seeds the optimizer with your current prompt and an FPR-aware feedback metric, so it evolves the prompt to stop over-blocking, then re-deploys the improved instruction.

## The bypass you'll find

Naive detectors — even small classifiers — collapse on simple obfuscation. Space out the characters (`i g n o r e   y o u r   r u l e s`) or base64-encode the payload and detection accuracy can fall off a cliff. The notebook demonstrates this and shows the fix: **normalize before judging** (collapse spacing, decode blobs) and feed both raw and normalized text to the judge.

## Takeaways

- Track **false-positive rate**, not just recall.
- **Normalize** inputs before detection.
- **Layer** endpoint guardrails (safety/PII) with an LLM judge (injection technique).
- **Version** every prompt change as an MLflow run and regression-test it.
- **Align** the judge to your policy with **DSPy + GEPA** — optimize from your current prompt, punishing false positives.

A guardrail you haven't evaluated is a guess. Make it a measurement → open the [notebook](guardrail_evaluation.ipynb).
