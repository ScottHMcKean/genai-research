"""Shared PII detection logic for the pii_serving demo.

Lifted from vllm/pii_detection_profiling.ipynb so every notebook uses the same schema,
system prompt, ground-truth parsing, masking and scoring. Kept dependency-light (pydantic
+ stdlib) so it imports cleanly in both notebook and serving contexts.
"""

from __future__ import annotations

import ast
import json
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# Output schema (constrains vLLM guided decoding) -- only has_pii + entities.
# masked_text is derived programmatically from the entity list (keeps output tokens small).
# --------------------------------------------------------------------------------------
class PIIType(str, Enum):
    PERSON = "PERSON"
    SSN = "SSN"
    ADDRESS = "ADDRESS"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    OTHER = "OTHER"


class PIIEntity(BaseModel):
    entity_type: PIIType
    original_value: str = Field(description="Exact text span copied verbatim from the document")
    salt_id: str = Field(description="Deterministic placeholder, e.g. PERSON_001, SSN_001")


class PIIDetectionResult(BaseModel):
    has_pii: bool
    entities: list[PIIEntity]


def pii_schema() -> dict:
    """JSON schema passed to vLLM guided decoding via extra_body={'guided_json': ...}."""
    return PIIDetectionResult.model_json_schema()


# --------------------------------------------------------------------------------------
# Ground-truth parsing (gretel pii_spans -> typed entities)
# --------------------------------------------------------------------------------------
GRETEL_TO_PII_TYPE = {
    "name": "PERSON", "first_name": "PERSON", "last_name": "PERSON",
    "ssn": "SSN", "social_security_number": "SSN",
    "street_address": "ADDRESS", "address": "ADDRESS", "city": "ADDRESS",
    "state": "ADDRESS", "zip_code": "ADDRESS",
    "email": "EMAIL", "email_address": "EMAIL",
    "phone_number": "PHONE", "phone": "PHONE",
    "credit_card_number": "CREDIT_CARD", "credit_card": "CREDIT_CARD",
    "bank_routing_number": "BANK_ACCOUNT", "iban": "BANK_ACCOUNT",
    "account_number": "BANK_ACCOUNT", "bank_account": "BANK_ACCOUNT",
    "date_of_birth": "DATE_OF_BIRTH", "dob": "DATE_OF_BIRTH",
}


def parse_ground_truth(pii_spans_raw, input_text):
    """Extract ground-truth PII entities from the dataset's pii_spans column."""
    try:
        if isinstance(pii_spans_raw, str):
            spans = json.loads(pii_spans_raw)
        elif isinstance(pii_spans_raw, list):
            spans = pii_spans_raw
        else:
            return []
    except (json.JSONDecodeError, ValueError):
        try:
            spans = ast.literal_eval(pii_spans_raw)
        except Exception:
            return []

    entities = []
    for span in spans:
        if isinstance(span, dict):
            label = span.get("label", span.get("type", "")).lower().strip()
            value = span.get("text", span.get("value", ""))
            if not value and "start" in span and "end" in span:
                value = input_text[span["start"]:span["end"]]
            if value and len(value.strip()) > 1:
                pii_type = GRETEL_TO_PII_TYPE.get(label, "OTHER")
                entities.append({"type": pii_type, "value": value.strip()})
    return entities


# --------------------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------------------
def build_masked_text(input_text, entities):
    """Replace each entity's original_value with its salt_id.

    entities may be PIIEntity objects or dicts with original_value/salt_id.
    Longest-first replacement avoids partial-overlap corruption.
    """
    def _val(e):
        return e.original_value if hasattr(e, "original_value") else e["original_value"]

    def _salt(e):
        return e.salt_id if hasattr(e, "salt_id") else e["salt_id"]

    masked = input_text
    for e in sorted(entities, key=lambda x: len(_val(x)), reverse=True):
        masked = masked.replace(_val(e), _salt(e))
    return masked


# --------------------------------------------------------------------------------------
# Scoring (entity-level precision / recall / F1)
# --------------------------------------------------------------------------------------
def normalize(s):
    return s.lower().strip().replace("-", "").replace(" ", "")


def score_document(predicted, ground_truth):
    """Entity-level precision/recall/F1 for one document.

    A predicted entity is a TP if its normalized value is a substring of (or contains) a
    ground-truth entity of the same type (OTHER matches any type).
    """
    if not ground_truth and not predicted:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    gt_matched = [False] * len(ground_truth)
    tp = 0
    fp = 0
    for pred in predicted:
        pred_norm = normalize(pred["value"])
        matched = False
        for j, gt in enumerate(ground_truth):
            if gt_matched[j]:
                continue
            gt_norm = normalize(gt["value"])
            type_match = pred["type"] == gt["type"] or gt["type"] == "OTHER"
            value_match = (pred_norm in gt_norm or gt_norm in pred_norm or pred_norm == gt_norm)
            if value_match and type_match:
                gt_matched[j] = True
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------------------
# System prompt (few-shot, verbatim-extraction instructions)
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a PII detection engine. Given a document, extract every personally \
identifiable information (PII) entity. Return valid JSON only.

## PII categories

PERSON -- Full names ("John Smith"), first names ("Sarah"), last names \
("Williams"), titles with names ("Dr. Maria Garcia", "Mr. James Lee"). \
Report each distinct person as one entity; combine first + last when adjacent.

SSN -- Social Security Numbers in any format: "123-45-6789", "123 45 6789", \
"123456789".

ADDRESS -- Street addresses, city/state/ZIP, or full mailing addresses. \
Combine contiguous components into one entity: "742 Evergreen Terrace, \
Springfield, IL 62704". Individual city names or ZIP codes that appear alone \
are also ADDRESS entities.

EMAIL -- Email addresses: "user@example.com".

PHONE -- Phone numbers in any format including country codes: \
"(555) 123-4567", "+1-800-555-0199", "555.123.4567".

CREDIT_CARD -- Credit/debit card numbers: "4111-1111-1111-1111", \
"5500 0000 0000 0004".

BANK_ACCOUNT -- Bank account numbers, routing numbers, IBANs, SWIFT/BIC \
codes: "Account #123456789", "Routing: 021000021", \
"IBAN: DE89370400440532013000".

DATE_OF_BIRTH -- Dates of birth in any format: "01/15/1990", \
"March 3, 1985", "1990-01-15", "born on May 12, 1978".

OTHER -- Any PII not covered above (passport numbers, driver's license \
numbers, tax IDs, national ID numbers).

## Rules

1. Extract EVERY occurrence. Missing an entity is worse than a false positive.
2. `original_value` must be the EXACT text span from the document -- copy \
verbatim, do not reformat or paraphrase.
3. Number salt_ids sequentially within each type: PERSON_001, PERSON_002, \
SSN_001, etc.
4. If the same value appears multiple times, reuse its salt_id.
5. For compound addresses (street + city + state + ZIP on the same line or \
sentence), report as one ADDRESS entity.
6. First name and last name that appear together are one PERSON entity; if \
only a first or last name appears alone elsewhere, report it separately.
7. Partial card/account numbers ("ending in 4532") are still BANK_ACCOUNT \
or CREDIT_CARD entities.

## Example 1

Input:
Dear Mr. James Wilson, your account ending in 4532 at 742 Evergreen \
Terrace, Springfield, IL 62704 has been flagged. Please contact us at \
support@finbank.com or call (312) 555-0198. SSN on file: 287-65-4321. \
Date of birth: 04/12/1983.

Output:
{"has_pii": true, "entities": [\
{"entity_type": "PERSON", "original_value": "James Wilson", \
"salt_id": "PERSON_001"}, \
{"entity_type": "BANK_ACCOUNT", "original_value": "4532", \
"salt_id": "BANK_ACCOUNT_001"}, \
{"entity_type": "ADDRESS", "original_value": "742 Evergreen Terrace, \
Springfield, IL 62704", "salt_id": "ADDRESS_001"}, \
{"entity_type": "EMAIL", "original_value": "support@finbank.com", \
"salt_id": "EMAIL_001"}, \
{"entity_type": "PHONE", "original_value": "(312) 555-0198", \
"salt_id": "PHONE_001"}, \
{"entity_type": "SSN", "original_value": "287-65-4321", \
"salt_id": "SSN_001"}, \
{"entity_type": "DATE_OF_BIRTH", "original_value": "04/12/1983", \
"salt_id": "DATE_OF_BIRTH_001"}]}

## Example 2

Input:
ACCOUNT STATEMENT -- Prepared for Elena Rodriguez (DOB: 1990-06-22). \
Mailing address: 1800 K Street NW, Washington, DC 20006. \
Primary phone: +1-202-555-0147. Email: e.rodriguez@globalmail.net. \
Routing number 091000019, account 00112233445. \
Visa ending 8821. SSN: 601-33-8877.

Output:
{"has_pii": true, "entities": [\
{"entity_type": "PERSON", "original_value": "Elena Rodriguez", \
"salt_id": "PERSON_001"}, \
{"entity_type": "DATE_OF_BIRTH", "original_value": "1990-06-22", \
"salt_id": "DATE_OF_BIRTH_001"}, \
{"entity_type": "ADDRESS", "original_value": "1800 K Street NW, \
Washington, DC 20006", "salt_id": "ADDRESS_001"}, \
{"entity_type": "PHONE", "original_value": "+1-202-555-0147", \
"salt_id": "PHONE_001"}, \
{"entity_type": "EMAIL", "original_value": "e.rodriguez@globalmail.net", \
"salt_id": "EMAIL_001"}, \
{"entity_type": "BANK_ACCOUNT", "original_value": "091000019", \
"salt_id": "BANK_ACCOUNT_001"}, \
{"entity_type": "BANK_ACCOUNT", "original_value": "00112233445", \
"salt_id": "BANK_ACCOUNT_002"}, \
{"entity_type": "CREDIT_CARD", "original_value": "8821", \
"salt_id": "CREDIT_CARD_001"}, \
{"entity_type": "SSN", "original_value": "601-33-8877", \
"salt_id": "SSN_001"}]}"""


# Compact prompt (no few-shot examples) -- used to FINE-TUNE and SERVE the LoRA model, so its training
# sequences fit an A10 (the full few-shot SYSTEM_PROMPT is ~1.5k tokens and OOMs the loss logits).
# The base models (Qwen, Gemma) keep the full few-shot SYSTEM_PROMPT for best zero-shot quality.
COMPACT_SYSTEM_PROMPT = """\
You are a PII detection engine. Extract every personally identifiable information entity from the \
document and return valid JSON only, matching {"has_pii": bool, "entities": [{"entity_type", \
"original_value", "salt_id"}]}.
Categories: PERSON, SSN, ADDRESS, EMAIL, PHONE, CREDIT_CARD, BANK_ACCOUNT, DATE_OF_BIRTH, OTHER.
Rules: extract EVERY occurrence; original_value must be the EXACT verbatim span from the document; \
number salt_ids sequentially per type (PERSON_001, PERSON_002, SSN_001, ...); reuse a salt_id if the \
same value repeats; combine contiguous address components (street, city, state, ZIP) into one ADDRESS."""


def build_messages(text, max_chars=2000, system_prompt=SYSTEM_PROMPT):
    """Chat messages for one document (shared system prefix enables vLLM prefix caching)."""
    if len(text) > max_chars:
        text = text[:max_chars]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract all PII entities from this document:\n\n{text}"},
    ]


def extract_pii(client, endpoint, text, max_tokens=500, temperature=0.1, system_prompt=SYSTEM_PROMPT):
    """Call an OpenAI-compatible serving endpoint with guided-JSON decoding.

    Returns (PIIDetectionResult | None, completion_tokens). None on parse failure.
    """
    resp = client.chat.completions.create(
        model=endpoint,
        messages=build_messages(text, system_prompt=system_prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"guided_json": pii_schema()},
    )
    n_tokens = resp.usage.completion_tokens if resp.usage else 0
    raw = resp.choices[0].message.content
    try:
        return PIIDetectionResult.model_validate_json(raw), n_tokens
    except Exception:
        return None, n_tokens
