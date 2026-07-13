# Config for the Document Intelligence demo -- reads the common FINS data.
# Matches the other use-case folders. To adapt to another discipline, point
# CATALOG/SCHEMA at your own and re-stage source PDFs via fins_data/generate_data.py.

CATALOG = "shm_skunkworks_catalog"    # <- change to your catalog
SCHEMA = "claims_demo"                # <- change to your schema
VOLUME = "raw_pdfs"                   # real public insurance PDFs, staged by fins_data/generate_data.py
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Tables this demo creates in the common schema
PARSED_DOCS = f"{CATALOG}.{SCHEMA}.parsed_docs"
PARSED_TEXT = f"{CATALOG}.{SCHEMA}.parsed_text"
EXTRACTED_FIELDS = f"{CATALOG}.{SCHEMA}.extracted_fields"
EXTRACTED_FLAT = f"{CATALOG}.{SCHEMA}.extracted_flat"
GOLDEN_LABELS = f"{CATALOG}.{SCHEMA}.golden_labels"

JUDGE_ENDPOINT = "databricks-claude-sonnet-4-5"
