import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load your trained model
model = joblib.load("medical_rf_model.pkl")

# Define input schema using Pydantic
class PatientData(BaseModel):
    age: int
    sex: int
    bp: float
    chol: float
    fbs: float
    restecg: int
    exng: int
    temperature: float
    o2: float
    hr: float

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def main():
    return {
        "message": "Welcome to ML-based Health Monitoring API"
    }

@app.post("/predict/")
def predict(data: PatientData):
    # Convert incoming data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Run prediction
    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction),
        "status": "high risk" if prediction == 1 else "normal"
    }
