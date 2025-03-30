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

@app.post("/predict")
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
