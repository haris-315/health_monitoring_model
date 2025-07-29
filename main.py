import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json

# Load the trained model
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def main():
    return {"message": "Welcome to ML-based Health Monitoring API"}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            try:
                # Parse incoming JSON data
                patient_data = json.loads(data)
                # Validate data using Pydantic
                patient = PatientData(**patient_data)
                # Convert to DataFrame
                input_df = pd.DataFrame([patient.dict()])
                # Run prediction
                prediction = model.predict(input_df)[0]
                # Send prediction back to client
                await websocket.send_json({
                    "prediction": int(prediction),
                    "status": "high risk" if prediction == 1 else "normal"
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()
