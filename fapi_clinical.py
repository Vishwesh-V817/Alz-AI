from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
from clinical import ClinicalAlzheimerInference

app = FastAPI(title="Clinical Alzheimer's Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL: Filename must match what your trainer produces
MODEL_PATH = "Decision_Tree_model.pkl"

engine = None
if os.path.exists(MODEL_PATH):
    try:
        engine = ClinicalAlzheimerInference(MODEL_PATH)
        print(f"✓ Clinical model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
else:
    print(f"WARNING: {MODEL_PATH} not found!")

class PatientData(BaseModel):
    Age: int = Field(..., ge=40, le=100)
    Gender: int; Ethnicity: int; EducationLevel: int; BMI: float
    Smoking: int; AlcoholConsumption: float; PhysicalActivity: float
    DietQuality: float; SleepQuality: float; FamilyHistoryAlzheimers: int
    CardiovascularDisease: int; Diabetes: int; Depression: int
    HeadInjury: int; Hypertension: int; SystolicBP: float; DiastolicBP: float
    CholesterolTotal: float; CholesterolLDL: float; CholesterolHDL: float
    CholesterolTriglycerides: float; MMSE: float; FunctionalAssessment: float
    ADL: float; MemoryComplaints: int; BehavioralProblems: int
    Confusion: int; Disorientation: int; PersonalityChanges: int
    DifficultyCompletingTasks: int; Forgetfulness: int

@app.post("/predict-clinical")
async def predict_clinical(patient: PatientData):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    result = engine.predict(patient.dict())
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)