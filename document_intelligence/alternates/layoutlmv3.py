# Databricks notebook source
# MAGIC %md
# MAGIC Experiment notebook for using layoutlmv3 to parse documents

# COMMAND ----------

from PIL import Image
from pathlib import Path

# Define the path to the image
image_path = Path("/Volumes/shm/multimodal/converted_images/D8400-APM_p0001.jpg")

# Load the image
image = Image.open(image_path)

# Display the image
display(image)

# COMMAND ----------

from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
import torch

# COMMAND ----------

from transformers import LayoutLMv3FeatureExtractor
from transformers import AutoProcessor
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

model = LayoutLMv3ForSequenceClassification.from_pretrained(
  "microsoft/layoutlmv3-base", 
  num_labels=5
  )

# Number of labels are the categories. Might be worth pulling a pre-trained model on the DocLayNet dataset to enforce consistent labels

# COMMAND ----------

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]


# COMMAND ----------

from datasets import load_dataset
dataset = load_dataset("ds4sd/DocLayNet", streaming=True)
subset = next(dataset)

# COMMAND ----------

from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
import numpy as np

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = AutoModelForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-invoice")

print(type(image))
width, height = image.size

# encode
encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
offset_mapping = encoding.pop('offset_mapping')

# forward pass
outputs = model(**encoding)

# get predictions
predictions = outputs.logits.argmax(-1).squeeze().tolist()
token_boxes = encoding.bbox.squeeze().tolist()

# only keep non-subword predictions
is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]


# COMMAND ----------

encoding = feature_extractor(image, return_tensors="pt")

# COMMAND ----------

encoding

# COMMAND ----------



# COMMAND ----------

with torch.no_grad():
    outputs = model(encoding)

predicted_labels = outputs.logits.argmax(-1).squeeze().tolist()

# COMMAND ----------

def scale_bounding_box(box, width_scale, height_scale):
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale),
    ]

width, height = image.size
width_scale = 1000 / width
height_scale = 1000 / height

# Assume you have OCR results in a list of dictionaries
ocr_results = [{"word": "example", "bounding_box": [10, 20, 30, 40]}, ...]

words = [result["word"] for result in ocr_results]
boxes = [scale_bounding_box(result["bounding_box"], width_scale, height_scale) for result in ocr_results]

# COMMAND ----------

from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from PIL import Image
import json
from pathlib import Path
import torch

# Initialize the processor
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

# Load the model
model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=2)

def scale_bounding_box(box, width_scale, height_scale):
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale),
    ]

def predict_document_image(image_path, model, processor):
    # Load OCR results (assuming they're saved in a JSON file)
    json_path = image_path.with_suffix(".json")
    with json_path.open("r") as f:
        ocr_result = json.load(f)
    
    # Open and process the image
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        width_scale = 1000 / width
        height_scale = 1000 / height
        
        words = []
        boxes = []
        for row in ocr_result:
            words.append(row["word"])
            boxes.append(scale_bounding_box(row["bounding_box"], width_scale, height_scale))
        
        # Prepare inputs for the model
        encoding = processor(image, words, boxes=boxes, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**encoding)
        
        # Get the predicted class
        predicted_class = outputs.logits.argmax().item()
        
    return predicted_class


# COMMAND ----------



# COMMAND ----------

#Download model from huggingface

# COMMAND ----------

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download the model from Hugging Face
model_path = hf_hub_download(repo_id="hantian/yolo-doclaynet", filename="yolov11m-doclaynet.pt")

# Load the downloaded model using Ultralytics YOLO
model = YOLO(model_path)

# Perform object detection on an image
results = model(image)

# Display the results
results[0].show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Assuming 'results' contains your detection results
img = results[0].plot()  # This returns a numpy array
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------


