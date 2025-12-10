import os
from io import BytesIO
from urllib import request

import numpy as np
import onnxruntime as ort
from PIL import Image
from keras_image_helper import create_preprocessor


model_name = "hair_classifier_empty.onnx"


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    x = np.array(img)
    return x


def preprocess_pytorch_style(X):
    # X: shape (1, 200, 200, 3), dtype=float32, values in [0, 255]
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW
    # from (batch, height, width, channels) → (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)  

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)


session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"]
)

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name


def lambda_handler(event, context):
    url = event["url"]
    img = download_image(url)
    x = prepare_image(img, target_size=(200, 200))
    X = preprocess_pytorch_style(np.array([x]))
    model_results = session.run([output_name], {input_name: X})
    prediction = model_results[0][0].tolist()
    result = {"result": prediction}
    return result
