# Databricks notebook source
# MAGIC %md
# MAGIC This notebook converts the PDF to images for easier testing. This would be wrapped in the production pipeline

# COMMAND ----------

# MAGIC %sh 
# MAGIC apt-get update
# MAGIC apt-get -f -y install poppler-utils

# COMMAND ----------

# MAGIC %pip install pdf2image
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pdf2image import convert_from_path
from pathlib import Path

# Convert PDF to images
pdf_path = Path("/Volumes/shm/multimodal/raw_documents/D8400-APM.pdf")
images = convert_from_path(pdf_path)

# COMMAND ----------

# Save each page as a separate image
for i, image in enumerate(images):
    print(f'Saving {pdf_path.stem} Page {i+1}')
    image.save(
      pdf_path.parent.parent 
      / 'converted_images' 
      / f"{pdf_path.stem}_p{i+1:04d}.jpg"
      , "JPEG")
