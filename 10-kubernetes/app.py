import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from keras_image_helper import create_preprocessor
from pydantic import BaseModel, Field


app = FastAPI(title="clothing-prediction")

model_name = "clothing-model.onnx"


class Request(BaseModel):
    url: str = Field(..., title="Image URL", example="http://bit.ly/mlbookcamp-pants")


class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float


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


preprocessor = create_preprocessor(preprocess_pytorch_style, target_size=(224, 224))


session = ort.InferenceSession(model_name, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name


classes = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]


@app.post("/predict")
def predict(request: Request) -> PredictResponse:
    X = preprocessor.from_url(request.model_dump()["url"])
    model_results = session.run([output_name], {input_name: X})
    predictions = model_results[0][0].tolist()

    predictions_dict = dict(zip(classes, predictions))
    
    top_class = max(predictions_dict, key=predictions_dict.get)
    top_probability = predictions_dict[top_class]

    return PredictResponse(
        predictions=predictions_dict,
        top_class=top_class,
        top_probability=top_probability
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
