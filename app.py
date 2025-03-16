import joblib
from fastapi import FastAPI # type: ignore
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from keras import losses
from sklearn.decomposition import PCA
from xgboost import XGBRegressor # type: ignore

xgb_model = joblib.load('/app/xgboost_model.pkl')
# Initialize PCA with 448 components
# Load the PCA model (assuming you've saved the PCA model during training)
pca = joblib.load('/app/pca_model_best.pkl')  # Load the saved PCA model

# FastAPI setup
app = FastAPI()

# Request body model
class SpectrumRequest(BaseModel):
    spectrum: list

@app.get("/")
def home():
    return {"message": "Welcome to the Prediction API!"}

@app.post("/predict")
def predict(request: SpectrumRequest):
    # Extract the spectrum from the request
    new_spectrum = np.array(request.spectrum)
    
    # Ensure the spectrum has the correct number of features (448)
    if new_spectrum.shape[0] > 448:
        new_spectrum = new_spectrum[:448]  # Truncate the input to 448 features if it's larger
    
    # Apply PCA to reduce the features to 448 (if the model is already trained with PCA)
    transformed_spectrum = pca.transform(new_spectrum.reshape(1, -1))
    
    # Make the prediction using the neural network model
    prediction = xgb_model.predict(transformed_spectrum)
    
    # Convert the numpy float32 prediction to a regular Python float
    prediction_value = float(prediction[0])  # or prediction[0] depending on the output

    return {"prediction": prediction_value}