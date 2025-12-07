import os

import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor


model_name = os.getenv("MODEL_NAME", "clothing_classifier_mobilenet_v2.onnx")


def preprocess_pytorch_style(X):
    # X: shape (1, 299, 299, 3), dtype=float32, values in [0, 255]
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW
    # from (batch, height, width, channels) → (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)  

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)


preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)


session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"]
)

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]


def lambda_handler(event, context):
    url = event["url"]
    X = preprocessor.from_url(url)
    model_results = session.run([output_name], {input_name: X})
    predictions = model_results[0][0].tolist()
    result = dict(zip(classes, predictions))
    return result

