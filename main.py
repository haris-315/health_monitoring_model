import joblib
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import httpx  # Async HTTP client

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
            data = await websocket.receive_text()
            try:
                patient_data = json.loads(data)
                patient = PatientData(**patient_data)
                input_df = pd.DataFrame([patient.dict()])
                prediction = model.predict(input_df)[0]
                await websocket.send_json({
                    "prediction": int(prediction),
                    "status": "high risk" if prediction == 1 else "normal"
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        try:
            await websocket.close()
        except:
            pass

# 👇 Background keep-alive pinger
async def keep_alive():
    await asyncio.sleep(5)  # Wait until server is fully up
    while True:
        try:
            async with httpx.AsyncClient() as client:
                # Replace with your actual Render URL (no trailing slash)
                response = await client.get("https://health-monitoring-model.onrender.com/")
                print(f"Keep-alive ping: {response.status_code}")
        except Exception as e:
            print(f"Keep-alive error: {e}")
        await asyncio.sleep(200)  # Ping every 20 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_alive())
