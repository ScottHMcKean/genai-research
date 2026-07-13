# Databricks notebook source
# MAGIC %md
# MAGIC Experiment notebook for using yolo to parse documents. 

# COMMAND ----------

# MAGIC %pip install ultralytics
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from PIL import Image
from pathlib import Path

# Define the path to the image
tbl_img_path = Path("/Volumes/shm/multimodal/converted_images/D8400-APM_p0029.jpg")
fig_img_path = Path("/Volumes/shm/multimodal/converted_images/D8400-APM_p0043.jpg")

# Load the image
tbl_image = Image.open(tbl_img_path)
fig_image = Image.open(fig_img_path)

# Display the image
display(tbl_image)
display(fig_image)

# COMMAND ----------

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face
model_path = hf_hub_download(
  repo_id="hantian/yolo-doclaynet", 
  filename="yolov11m-doclaynet.pt"
  )

# Load the downloaded model using Ultralytics YOLO
model = YOLO(model_path)

# Perform object detection on an image
tbl_results = model(tbl_image, iou=0.4, conf=0.25)
fig_results = model(fig_image, iou=0.4, conf=0.25)

# Display the results
display(tbl_results[0])

# COMMAND ----------

import matplotlib.pyplot as plt
img = tbl_results[0].plot()
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
img = fig_results[0].plot()
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------

import shutil
shutil.rmtree('./yolo_outputs')
tbl_results[0].save_crop('./yolo_outputs/tbl')
tbl_results[0].save_txt('./yolo_outputs/tbl/yolo_text.txt')
fig_results[0].save_crop('./yolo_outputs/fig')
fig_results[0].save_txt('./yolo_outputs/fig/yolo_text.txt')

# COMMAND ----------

# we will need a seperate OCR pipeline in addition to image extraction.
import pytesseract

tbl_string = pytesseract.image_to_string(tbl_image)
with open('./yolo_outputs/tbl/ocr_text.txt', 'w') as file:
    file.write(tbl_string)

fig_string = pytesseract.image_to_string(fig_image)
with open('./yolo_outputs/fig/ocr_text.txt', 'w') as file:
    file.write(fig_string)

# COMMAND ----------


