# Databricks notebook source
# MAGIC %md
# MAGIC # FINS data — generate the common demo dataset
# MAGIC
# MAGIC **The single script** that builds the one Unity Catalog schema every use-case folder
# MAGIC reads from (`agents/`, `governance/`, `ai_runtime/`, `document_intelligence/`). Runs on
# MAGIC **serverless**, no extra installs. Idempotent — safe to re-run.
# MAGIC
# MAGIC Run this once, then run any use-case demo. To adapt to another discipline, swap the
# MAGIC generators below for your domain and update `config.py`.
# MAGIC
# MAGIC Creates in `shm_skunkworks_catalog.claims_demo`:
# MAGIC
# MAGIC | Object | Feeds |
# MAGIC |---|---|
# MAGIC | `claims` (structured, incl. PII) | Genie space, PII guardrails, fine-tune labels |
# MAGIC | `claim_notes` (free-text adjuster notes + severity) | RAG, fine-tune text |
# MAGIC | `/Volumes/.../docs` (policy wordings + handling guidelines, `.md`) | Knowledge Assistant, RAG index |
# MAGIC | `doc_chunks` (chunked docs, CDF on) | Vector Search Delta-Sync index |
# MAGIC
# MAGIC All synthetic — no real customer data.

# COMMAND ----------

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) if "__file__" in dir() else ".")
# config.py sits next to this notebook
sys.path.insert(0, ".")
sys.path.insert(0, os.getcwd())
from config import (
    CATALOG, SCHEMA, VOLUME, VOLUME_PATH, CLAIMS_TABLE, NOTES_TABLE,
    CHUNKS_TABLE, N_CLAIMS, SEED,
)

# Catalog already exists on the FEVM (no metastore-level CREATE CATALOG needed).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print("Schema + volume ready:", VOLUME_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Structured claims  (→ Genie, PII guardrails, fine-tune labels)
# MAGIC
# MAGIC Synthetic P&C claims with intentional **PII columns** (`customer_name`,
# MAGIC `customer_email`, `customer_ssn`) so the governance demo has something to mask.
# MAGIC `severity` is the label the fine-tuning demo learns to predict from the note text.

# COMMAND ----------

import random
from datetime import date, timedelta

rnd = random.Random(SEED)

FIRST = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","David","Elizabeth",
         "Priya","Wei","Ahmed","Sofia","Liam","Emma","Noah","Olivia","Lucas","Chloe"]
LAST  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Patel","Nguyen",
         "Chen","Singh","Kim","Lopez","Tremblay","Roy","Gagnon","Cote","MacDonald","Wong"]
REGIONS = ["Ontario","Quebec","British Columbia","Alberta","Nova Scotia","Manitoba"]
ADJUSTERS = ["A. Okafor","B. Lévesque","C. Thompson","D. Sharma","E. Rossi","F. Dubois"]

# claim_type -> (perils, [low_templates], [med_templates], [high_templates])
CLAIM_TYPES = {
    "Auto": {
        "perils": ["Collision","Theft","Hail","Vandalism","Windshield"],
        "low":  ["Minor parking-lot scrape, cosmetic bumper damage, vehicle drivable, no injuries.",
                 "Cracked windshield from road debris. No other damage. Glass-only claim.",
                 "Small hail dents on hood and roof. Cosmetic only, mechanically sound."],
        "medium":  ["Rear-end collision at intersection, moderate rear-quarter damage, airbags not deployed, towed for assessment.",
                 "Vehicle broken into overnight, stolen electronics and interior damage, police report filed.",
                 "Side-swipe on highway, two panels and mirror damaged, driver reports minor whiplash."],
        "high": ["Multi-vehicle collision on 401, front end destroyed, driver hospitalized, potential total loss and third-party injury.",
                 "Vehicle stolen and not recovered, financed loan outstanding, total-loss settlement required.",
                 "Rollover accident, extensive structural damage, two occupants transported to hospital, liability disputed."],
    },
    "Property": {
        "perils": ["Water","Fire","Wind","Theft","Sewer Backup"],
        "low":  ["Small kitchen water leak under sink, localized cabinet damage, dried within a day.",
                 "Fence panels blown down in wind storm, no structural damage to dwelling.",
                 "Minor smoke residue from a stovetop flare-up, cleaned, no structural fire damage."],
        "medium":  ["Basement flooding from sewer backup, flooring and drywall damage across two rooms, mitigation crew engaged.",
                 "Wind-driven rain through damaged roof, ceiling and insulation damage in upper bedroom.",
                 "Kitchen fire contained to one room, smoke damage throughout main floor, temporary relocation needed."],
        "high": ["House fire, significant structural damage to two floors, dwelling uninhabitable, ALE and rebuild required.",
                 "Major flood, standing water throughout main floor, mold risk, extensive contents loss.",
                 "Severe wind storm removed section of roof, water intrusion throughout, dwelling unsafe to occupy."],
    },
    "Liability": {
        "perils": ["Slip and Fall","Dog Bite","Property Damage"],
        "low":  ["Guest slipped on walkway, no medical treatment sought, precautionary report only.",
                 "Minor property damage to neighbour's fence, small settlement expected."],
        "medium":  ["Visitor fell on icy steps, sprained wrist, sought medical attention, claim under premises liability.",
                 "Dog bite incident, puncture wounds requiring stitches, claimant seeking medical costs."],
        "high": ["Serious slip-and-fall, claimant hospitalized with fracture, litigation threatened, reserves elevated.",
                 "Liability claim with alleged permanent injury and lost wages, legal counsel engaged."],
    },
}
SEV_MIX = ["Low"]*45 + ["Medium"]*40 + ["High"]*15  # roughly realistic skew
STATUSES = ["Open","In Review","Approved","Denied","Closed"]

rows = []
notes = []
today = date(2026, 6, 30)
for i in range(N_CLAIMS):
    fn, ln = rnd.choice(FIRST), rnd.choice(LAST)
    name = f"{fn} {ln}"
    ctype = rnd.choice(list(CLAIM_TYPES))
    spec = CLAIM_TYPES[ctype]
    peril = rnd.choice(spec["perils"])
    sev = rnd.choice(SEV_MIX)
    note_text = rnd.choice(spec[sev.lower()])
    loss = today - timedelta(days=rnd.randint(1, 540))
    report = loss + timedelta(days=rnd.randint(0, 10))
    base = {"Low": (500, 8000), "Medium": (8000, 60000), "High": (60000, 500000)}[sev]
    amount = round(rnd.uniform(*base), 2)
    status = rnd.choice(STATUSES)
    paid = round(amount * rnd.uniform(0.4, 1.0), 2) if status in ("Approved","Closed") else 0.0
    claim_id = f"CLM-{100000+i}"
    rows.append((
        claim_id, f"POL-{rnd.randint(200000,299999)}", name,
        f"{fn.lower()}.{ln.lower()}@example.com",
        f"{rnd.randint(100,899)}-{rnd.randint(10,99)}-{rnd.randint(1000,9999)}",  # fake SSN-shaped PII
        ctype, peril, status, loss, report, amount, paid,
        rnd.choice(ADJUSTERS), rnd.choice(REGIONS), sev,
    ))
    notes.append((claim_id, ctype, peril, sev, note_text))

claims_cols = ["claim_id","policy_number","customer_name","customer_email","customer_ssn",
               "claim_type","peril","claim_status","loss_date","report_date",
               "claim_amount","paid_amount","adjuster_name","region","severity"]
df = spark.createDataFrame(rows, claims_cols)
df.write.mode("overwrite").option("overwriteSchema","true").saveAsTable(CLAIMS_TABLE)
spark.sql(f"COMMENT ON TABLE {CLAIMS_TABLE} IS 'Synthetic P&C insurance claims for the demo suite. Contains synthetic PII columns.'")
print(f"Wrote {df.count()} claims -> {CLAIMS_TABLE}")
display(spark.table(CLAIMS_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Adjuster notes  (→ RAG text, fine-tune training text)
# MAGIC
# MAGIC One free-text note per claim, paired with the `severity` label. The fine-tuning
# MAGIC demo learns note-text → severity (triage); the RAG demo can retrieve over them.

# COMMAND ----------

notes_df = spark.createDataFrame(notes, ["claim_id","claim_type","peril","severity","note"])
notes_df.write.mode("overwrite").option("overwriteSchema","true").saveAsTable(NOTES_TABLE)
print(f"Wrote {notes_df.count()} notes -> {NOTES_TABLE}")
display(spark.table(NOTES_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Knowledge documents  (→ Knowledge Assistant + RAG index)
# MAGIC
# MAGIC Short synthetic **policy wordings** and **claims-handling guidelines** written as
# MAGIC Markdown to the Volume. These are the authoritative docs the KA/RAG answers from.

# COMMAND ----------

DOCS = {
"auto_claims_handling_guideline.md": """# Auto Claims Handling Guideline

## Scope
Applies to all personal automobile claims (collision, theft, hail, vandalism, windshield).

## Triage & Severity
- **Low**: cosmetic-only, vehicle drivable, no injuries. Target close: 5 business days.
- **Medium**: moderate damage requiring assessment, possible minor injury, towing involved.
- **High**: potential total loss, hospitalization, third-party injury, or disputed liability. Escalate to senior adjuster within 24 hours.

## First Notice of Loss (FNOL)
Capture policy number, date/time and location of loss, police report number if applicable, and photos. Confirm coverage is in force on the loss date before proceeding.

## Windshield / Glass-only
Glass-only claims may be settled without a physical inspection when the estimate is under $1,000 and photos are provided.

## Total Loss
Declare total loss when repair cost exceeds 75% of actual cash value. Obtain two valuations and confirm any outstanding finance/lien before settlement.
""",

"property_water_damage_procedure.md": """# Property Claims — Water Damage Procedure

## Covered vs Excluded
Sudden and accidental water discharge (burst pipe, appliance failure) is covered. Gradual seepage, long-term leaks, and flood from surface water are **excluded** under the base policy; flood requires a separate endorsement. Sewer backup is covered **only** when the sewer-backup endorsement is present.

## Immediate Steps
1. Instruct the insured to stop the water source and prevent further damage (duty to mitigate).
2. Engage an approved mitigation vendor for water extraction and drying within 24 hours to limit mold risk.
3. Document moisture readings and photograph affected areas before drying.

## Reserves
Set initial reserves based on affected square footage. Basement finished-area losses commonly reach Medium severity; multi-floor standing water is High.
""",

"fire_claims_playbook.md": """# Fire Claims Playbook

## Safety First
Do not authorize re-entry until the fire department confirms the structure is safe. Board-up and securing services are covered as reasonable emergency measures.

## Additional Living Expenses (ALE)
When the dwelling is uninhabitable, ALE covers reasonable temporary housing and increased living costs up to the policy sublimit. Require receipts.

## Investigation
For any fire loss over $50,000 or with suspicious origin, engage a cause-and-origin investigator before settlement. Smoke-only losses without structural damage are typically Low–Medium.
""",

"fraud_indicators_checklist.md": """# Fraud Indicators Checklist

Escalate to the Special Investigations Unit (SIU) when three or more indicators are present:

- Loss occurs shortly after policy inception or a coverage increase.
- No police or fire report where one would be expected.
- Claimant is unusually familiar with claims process and pressures for fast settlement.
- Inconsistent or shifting description of the loss.
- Receipts are round-numbered, sequential, or lack vendor detail.
- Prior claims history with similar losses.

SIU referral does not deny the claim; it pauses settlement pending review. Document the specific indicators observed.
""",

"claims_escalation_matrix.md": """# Claims Escalation Matrix

| Trigger | Route to | SLA |
|---|---|---|
| Reserve > $100,000 | Senior adjuster + manager sign-off | 24 hours |
| Bodily injury / hospitalization | Injury/liability desk | 24 hours |
| Litigation threatened or served | Legal counsel | 48 hours |
| Suspected fraud (see checklist) | Special Investigations Unit | 3 business days |
| Catastrophe event (declared CAT) | CAT response team | Per CAT plan |

All escalations must be logged in the claim file with the trigger and the timestamp.
""",

"catastrophe_response_plan.md": """# Catastrophe (CAT) Response Plan

A CAT is declared when a single event (wind, wildfire, flood) is expected to generate a large volume of claims in a region.

## During a CAT
- Fast-track Low-severity claims with streamlined documentation to speed customer recovery.
- Prioritize High-severity and uninhabitable-dwelling claims for ALE and mitigation.
- Use mobile/virtual inspection where safe access is delayed.

## Reserving
Apply CAT reserve guidance and tag all related claims with the CAT event code for aggregate tracking.
""",

"personal_auto_policy_wording.md": """# Personal Automobile Policy — Key Wording (Summary)

## Coverages
- **Collision**: pays for direct and accidental damage to the insured vehicle from impact, less the deductible.
- **Comprehensive**: theft, fire, hail, vandalism, falling objects, and glass.
- **Third-Party Liability**: bodily injury and property damage the insured is legally liable for, up to the limit.

## Deductibles
The deductible shown on the declarations page applies per occurrence. Glass claims may carry a separate or waived deductible by endorsement.

## Exclusions
Racing, intentional damage, use for commercial delivery without endorsement, and wear-and-tear are excluded.
""",

"homeowners_policy_wording.md": """# Homeowners Policy — Key Wording (Summary)

## Coverages
- **Dwelling (Coverage A)**: the structure, at replacement cost when insured to value.
- **Contents (Coverage C)**: personal property, subject to special sublimits for jewellery, electronics, and cash.
- **Additional Living Expenses**: extra costs when the home is uninhabitable due to a covered loss.

## Endorsements
Flood (overland water) and sewer backup are **not** included by default and must be added by endorsement.

## Conditions
The insured must take reasonable steps to protect the property from further damage after a loss (duty to mitigate) and cooperate with the investigation.
""",
}

for fname, body in DOCS.items():
    with open(f"{VOLUME_PATH}/{fname}", "w") as f:
        f.write(body)
print(f"Wrote {len(DOCS)} knowledge docs -> {VOLUME_PATH}")
display(dbutils.fs.ls(VOLUME_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Chunk docs  (→ Vector Search source table, CDF enabled)
# MAGIC
# MAGIC Simple paragraph-ish chunking into a Delta table with Change Data Feed on, so the
# MAGIC Vector Search Delta-Sync index (built in the Agent Bricks demo) stays current.

# COMMAND ----------

import re
from pyspark.sql import Row

def chunk_text(text, max_chars=900):
    # split on markdown headers / blank lines, then pack to ~max_chars
    parts = re.split(r"\n(?=#)|\n\s*\n", text)
    chunks, cur = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

chunk_rows = []
cid = 0
for fname, body in DOCS.items():
    title = body.splitlines()[0].lstrip("# ").strip()
    for ci, ch in enumerate(chunk_text(body)):
        chunk_rows.append(Row(chunk_id=f"{fname}::{ci}", doc=fname, title=title, chunk=ch))
        cid += 1

chunks_df = spark.createDataFrame(chunk_rows)
(chunks_df.write.mode("overwrite").option("overwriteSchema","true")
    .option("delta.enableChangeDataFeed","true")
    .saveAsTable(CHUNKS_TABLE))
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Wrote {chunks_df.count()} chunks -> {CHUNKS_TABLE} (CDF on)")
display(spark.table(CHUNKS_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Real public insurance PDFs  (→ Document Intelligence demo)
# MAGIC
# MAGIC Downloads a handful of **genuine public** ACORD / CMS insurance PDFs into a Volume so
# MAGIC the `document_intelligence/` demo (parse → extract → evaluate) runs on the same common
# MAGIC schema. Real, messy layouts (OCR, multi-column, template placeholders).

# COMMAND ----------

import requests
from config import PDF_VOLUME, PDF_VOLUME_PATH

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{PDF_VOLUME}")
dbutils.fs.mkdirs(PDF_VOLUME_PATH)

REAL_DOCS = {
    "acord_certificate.pdf":      "https://azdot.gov/sites/default/files/2019/06/sample-insurance-certificate-accord-form.pdf",
    "acord_nyc_dcla.pdf":         "https://www.nyc.gov/assets/dcla/downloads/pdf/cdf_sample_cgl.pdf",
    "acord_nyc_mome.pdf":         "https://www.nyc.gov/assets/mome/pdf/Sample-MOME-ACORD_25_2016-03.pdf",
    "nyc_workerscomp_sample.pdf": "https://www.nyc.gov/html/dcla/downloads/pdf/cdf_sample_workerscomp_c_105_2.pdf",
    "acord_nyc_dycd.pdf":         "https://www.nyc.gov/assets/dycd/downloads/pdf/Insurance-Sample-Package25.pdf",
    "acord_moval.pdf":            "https://moval.gov/departments/media/pdf/SampleInsuranceCertsEndorsements.pdf",
    "acord_ny_ogs.pdf":           "https://ogs.ny.gov/system/files/documents/2018/10/acord25version918dcsamplenopollution.pdf",
    "nyc_acord_sample.pdf":       "https://www.nyc.gov/assets/buildings/pdf/acord_cert_of_ins_sample.pdf",
    "cms1500_claim_form.pdf":     "https://www.cms.gov/medicare/cms-forms/cms-forms/downloads/cms1500.pdf",
    "acord_tx.pdf":               "https://www.rrc.texas.gov/media/kh2p1ugz/acord25_-certificate-of-insurance.pdf",
}
ok = 0
for fname, url in REAL_DOCS.items():
    dest = f"{PDF_VOLUME_PATH}/{fname}"
    if os.path.exists(dest):
        ok += 1
        continue
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        ok += 1
    except Exception as e:
        print(f"  skip {fname}: {e}")
print(f"Staged {ok}/{len(REAL_DOCS)} insurance PDFs -> {PDF_VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Done — common FINS data spine is ready
# MAGIC
# MAGIC One script, one Unity Catalog schema. Every use-case folder (`agents/`,
# MAGIC `governance/`, `ai_runtime/`, `document_intelligence/`) reads from here.
# MAGIC
# MAGIC **To adapt to another discipline:** swap the synthetic generators + `DOCS` above for
# MAGIC your own domain (keep the same table/column shape, or update each folder's `config.py`).
