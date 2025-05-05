from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input  # EfficientNet-specific preprocessing

from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import RandomHeight, RandomWidth  # Import custom layers

app = FastAPI()

# Register the custom layers
get_custom_objects().update({'RandomHeight': RandomHeight, 'RandomWidth': RandomWidth})

# Now load your model inside a custom object scope
MODEL_PATH = "app/model_1.h5"

# Load the model with custom object scope
with tf.keras.utils.custom_object_scope({'RandomHeight': RandomHeight, 'RandomWidth': RandomWidth}):
    model = tf.keras.models.load_model(MODEL_PATH)

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, rescale=False):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    """
    # Decode it into a tensor
    img = tf.io.decode_image(filename, channels=3)  # Ensure 3 color channels
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    # Rescale the image
    if rescale:
        return img / 255.
    else:
        return img

# Define image size expected by EfficientNetB0
IMG_SIZE = (224, 224)

def preprocess_image(file: UploadFile) -> np.ndarray:
    """Preprocess the uploaded image for model prediction."""
    contents = file.file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")  # Ensure RGB format
    img = img.resize(IMG_SIZE)  # Resize to model's expected input size
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess input for EfficientNet
    return img_array

@app.post("/predict_cats_and_dogs")
async def predict(file: UploadFile = File(...)):
    """Predict the class of the uploaded image."""
    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
        confidence = float(np.max(predictions))  # Get confidence score

        # Define class labels (update these as per your dataset)
        class_labels = ["Cat", "Dog"]  # Example labels: Cat, Dog
        predicted_class_label = class_labels[predicted_class]

        return {"predicted_class": predicted_class_label, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}

import os
from fastapi.templating import Jinja2Templates

current_directory = os.getcwd()
templates = Jinja2Templates(directory=os.path.join(current_directory, "app", "templates"))

@app.get("/")
def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})



from fastapi import FastAPI, Form, Request, HTTPException
import joblib


# Available models
MODEL_PATHS = {
    "decision_tree": "ml_models/decision_tree.pkl",
    "svm": "ml_models/svm.pkl",
    "knn": "ml_models/knn.pkl",
    "logistic_regression": "ml_models/logistic_regression.pkl"
}

CLASS_LABELS = ["setosa", "versicolor", "virginica"]


@app.get("/iris")
def read_form(request: Request):
    return templates.TemplateResponse("iris.html", {"request": request})


@app.post("/predict_iris2")
async def predict(
    request: Request,
    model_name: str = Form(...),
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    try:
        # Load the selected model
        model_file = MODEL_PATHS.get(model_name)
        if not model_file or not os.path.exists(model_file):
            return {"error": f"Model '{model_name}' not found."}

        model = joblib.load(model_file)

        # Prepare input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        pred_class_index = model.predict(input_data)[0]
        pred_class_label = CLASS_LABELS[pred_class_index]

        # Get confidence
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(input_data)))
        else:
            confidence = 1.0  # fallback if not available

        return templates.TemplateResponse(
            "iris.html",
            {
                "request": request,
                "prediction": pred_class_label,
                "confidence": round(confidence * 100, 2),
                "selected_model": model_name
            }
        )

    except Exception as e:
        return {"error": str(e)}



# Determine the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full path to the ml_models directory
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

# Dynamically build the MODEL_PATHS dictionary
MODEL_PATHS = {}
if os.path.exists(MODEL_DIR):
    for model_file in os.listdir(MODEL_DIR):
        if model_file.endswith(".pkl"):
            model_name = os.path.splitext(model_file)[0]
            MODEL_PATHS[model_name] = os.path.join(MODEL_DIR, model_file)
else:
    raise FileNotFoundError(f"'ml_models' directory not found at expected location: {MODEL_DIR}")

# Class labels for the iris dataset
CLASS_LABELS = ["setosa", "versicolor", "virginica"]

@app.get("/iris")
def read_form(request: Request):
    return templates.TemplateResponse("iris.html", {"request": request})

@app.post("/predict_iris")
async def predict(
    request: Request,
    model_name: str = Form(...),
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    try:
        if model_name not in MODEL_PATHS:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

        model_file = MODEL_PATHS[model_name]
        model = joblib.load(model_file)

        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_class_index = model.predict(input_data)[0]
        pred_class_label = CLASS_LABELS[pred_class_index]

        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(input_data)))
        else:
            confidence = 1.0

        return templates.TemplateResponse(
            "iris.html",
            {
                "request": request,
                "prediction": pred_class_label,
                "confidence": round(confidence * 100, 2),
                "selected_model": model_name
            }
        )

    except Exception as e:
        return {"error": str(e)}