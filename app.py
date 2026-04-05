import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "cifar10_cnn_model.h5")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
ml_models = {}


def load_test_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()
    return x_test, y_test


def image_to_base64(img_array):
    img_uint8 = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["cnn"] = tf.keras.models.load_model(MODEL_PATH)
    ml_models["x_test"], ml_models["y_test"] = load_test_data()
    yield
    ml_models.clear()


app = FastAPI(title="CIFAR-10 Classifier", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict")
async def predict():
    x_test = ml_models["x_test"]
    y_test = ml_models["y_test"]
    model = ml_models["cnn"]

    image_index = int(np.random.randint(0, len(x_test)))
    image = x_test[image_index]
    true_label_index = int(y_test[image_index])

    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    predicted_label_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100.0)

    return JSONResponse(
        {
            "true_label": CLASS_NAMES[true_label_index],
            "predicted_label": CLASS_NAMES[predicted_label_index],
            "confidence": confidence,
            "image_base64": image_to_base64(image),
        }
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
