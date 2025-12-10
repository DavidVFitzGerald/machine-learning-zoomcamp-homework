import os
from io import BytesIO
from urllib import request

from PIL import Image
import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor


model_name = "hair_classifier_v1.onnx"


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
    return img


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

print(f"Length of inputs: {len(inputs)}")
print(f"Length of outputs: {len(outputs)}")

print("Names of inputs:")
for inp in inputs:
    print(inp.name)

print("Names of outputs:")
for out in outputs:
    print(out.name)

print(f"output_name: {output_name}")

classes = [
    "straight",
    "curly",
]

if __name__ == "__main__":
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
    x = np.array(img)
    X = np.array([x])
    print(f"Shape of X before preprocessing: {X.shape}")
    X = preprocess_pytorch_style(X)
    print(f"Shape of X after preprocessing: {X.shape}")
    print(f"Value of first pixel of red channel after preprocessing: {X[0,0,0,0]}")
    model_results = session.run([output_name], {input_name: X})
    type(model_results)
    print(model_results)
    predictions = model_results[0][0].tolist()
    result = dict(zip(classes, predictions))
    print(f"Model result: {result}")
    
