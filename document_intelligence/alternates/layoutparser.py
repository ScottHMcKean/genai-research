# Databricks notebook source
# MAGIC %md
# MAGIC Experiment notebook for using multimodal LLMs to parse documents and engineering drawings.

# COMMAND ----------

# MAGIC %pip install layoutparser pdf2image torchvision
# MAGIC %pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import layoutparser as lp
import cv2
import pdf2image

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

# MAGIC %md
# MAGIC LayoutParser seems to leverage region-based convolutional neural nets a lot, but hasn't extended to transformers from what I read.

# COMMAND ----------

# Load a pre-trained layout model
model = lp.Detectron2LayoutModel(
  'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
  extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
  label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
  )

# COMMAND ----------

import layoutparser as lp
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Convert BGR to RGB
fig_img_array = np.array(fig_image)
fig_img_array = fig_img_array[..., ::-1]  

# Detect the layout of the input image
layout = model.detect(fig_img_array)

# Visualize the detected layout
viz_image = lp.draw_box(fig_img_array, layout, box_width=3, box_alpha=0.1)
plt.figure(figsize=(12, 8))
plt.imshow(viz_image)
plt.title("Layout Analysis Result")
plt.axis('off')
plt.show()

# Process the detected layout
for block in layout:
    print(f"Type: {block.type}, Coordinates: {block.block}, Confidence: {block.score}")

# COMMAND ----------

# MAGIC %md
# MAGIC Extracting the different classes of blocks is relatively straightforward

# COMMAND ----------

text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
table_blocks = lp.Layout([b for b in layout if b.type=='Table'])

# COMMAND ----------

from PIL import Image
cropped_image = figure_blocks[0].crop_image(fig_img_array)
pil_image = Image.fromarray(cropped_image)
display(pil_image)

# COMMAND ----------

import layoutparser as lp
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Convert BGR to RGB
tbl_img_array = np.array(tbl_image)
tbl_img_array = tbl_img_array[..., ::-1]  

# Detect the layout of the input image
layout = model.detect(tbl_img_array)

# Visualize the detected layout
viz_image = lp.draw_box(tbl_img_array, layout, box_width=5, box_alpha=0.1, show_element_id=True)
plt.figure(figsize=(12, 8))
plt.imshow(viz_image)
plt.title("Layout Analysis Result")
plt.axis('off')
plt.show()

# Process the detected layout
for block in layout:
    print(f"Type: {block.type}, Coordinates: {block.block}, Confidence: {block.score}")
