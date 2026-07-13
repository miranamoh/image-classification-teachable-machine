# image-classification-teachable-machine
Cat vs Dog image classifier built with Teachable Machine, exported as Keras model, with a Python prediction script.
# Task Name: Image Recognition Model with Teachable Machine

This Task demonstrates training an image classification model using **Google's Teachable Machine**, exporting it in TensorFlow (Keras) format, and running predictions with a Python script.

## Overview

- **Tool used:** [Teachable Machine](https://teachablemachine.withgoogle.com/) by Google
- **Classes:** 2 classes — `Cat` and `Dog`
- **Export format:** TensorFlow → Keras (`.h5`)
- **Environment:** Google Colab

## Project Structure

```
.
├── keras_model.h5      # Trained model exported from Teachable Machine
├── labels.txt           # Class labels (Class 0: Cat, Class 1: Dog)
├──  outputP.png         # Screenshot of the script output (class + confidence score)
├── predict.py         # Python script that loads the model and predicts on an image
├── test1.png        # Sample test image used for evaluation
└── README.md
```

## How It Works

1. **Train the model:** Images of cats and dogs were uploaded to Teachable Machine, split into two classes, and trained directly in the browser.
2. **Export the model:** The trained model was exported using the *TensorFlow → Keras* option, producing a `keras_model.h5` file and a `labels.txt` file.
3. **Run predictions:** The `predict.py` script:
   - Loads the exported Keras model
   - Accepts an input image
   - Resizes/crops it to 224x224 and normalizes pixel values
   - Predicts the class and prints the confidence score

## Script Summary (`predict.py`)

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import tf_keras as tk

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tk.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Prepare the image array
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load and preprocess the image (replace with the path to your image)
image = Image.open("test1.png").convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

# Predict
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```
> **Note:** The original script (`predict.py`) was generated from a Google Colab notebook and includes Colab-specific lines (`from google.colab import files` and `uploaded = files.upload()`) used to upload the test image interactively. These lines only work inside Google Colab — if running the script locally, remove them and simply place the test image in the same folder as the script.

## Sample Output

Tested on a sample image of a cat:

```
Class: Cat
Confidence Score: 0.99967897
```

A screenshot of this output is included in the repository (`outputP.png`).

## Requirements

```
tensorflow
tf_keras
pillow
numpy
```

Install with:
```bash
pip install tensorflow tf_keras pillow numpy
```

## How to Run

```bash
python predict.py
```

Make sure `keras_model.h5`, `labels.txt`, and the image you want to test are in the same directory (or update the paths in the script).
