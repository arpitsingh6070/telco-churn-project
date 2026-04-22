from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI(title="Telco Customer Churn Prediction API")

# Setup CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema matching the frontend's expected data
class ChurnPredictionRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Load the trained model globally
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_model.pkl')
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"Error loading model from {model_path}: {e}")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Telco Customer Churn API is running"}

@app.post("/predict")
def predict_churn(request: ChurnPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Convert request to DataFrame
        data = request.model_dump()
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0][1] # Probability of Churn (class 1)
        
        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": float(prediction_proba)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
