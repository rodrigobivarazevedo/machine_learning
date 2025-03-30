from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the TensorFlow model (update path if needed)
#MODEL_PATH = "../saved_models/mobile_net_transfer_learning_cat_dog.h5"

MODEL_PATH = "app/mobile_net_transfer_learning_cat_dog.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size expected by your model
IMG_SIZE = (224, 224)  # Change if your model uses a different size

def preprocess_image(file: UploadFile) -> np.ndarray:
    """Preprocess the uploaded image for model prediction."""
    contents = file.file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")  # Ensure RGB format
    img = img.resize(IMG_SIZE)  # Resize to model's expected input size
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values (if required by your model)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of the uploaded image."""
    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
        confidence = float(np.max(predictions))  # Get confidence score
        
        return {"predicted_class": int(predicted_class), "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}

# Run the app using: uvicorn main:app --reload

# Define class labels
class_labels = ["Cat", "Dog"]  # Index 0 → Cat, Index 1 → Dog

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict if the image is a Cat or Dog."""
    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]  # Get class index
        confidence = float(np.max(predictions))  # Get confidence score

        # Map index to class name
        predicted_class = class_labels[predicted_index]

        return {"predicted_class": predicted_class, "confidence": confidence}
    
    except Exception as e:
        return {"error": str(e)}
