import pickle
from typing import Any

from fastapi import FastAPI
import uvicorn


app = FastAPI(title="subscription-prediction")


with open("pipeline_v2.bin", "rb") as f:
    pipeline = pickle.load(f)


@app.post("/predict")
def predict(record: dict[str, Any]) -> float:
    prob = pipeline.predict_proba([record])[0, 1]
    return prob


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)