from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import shutil
import os
import uvicorn

# --- Import Engines ---
from mri_infer import AlzheimerInferenceEngine
from drug import DrugRepurposingEngine
from clinical import ClinicalAlzheimerInference 

app = FastAPI(title="Integrated Alzheimer's AI Suite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Paths ---
MRI_MODEL_PATH = "models_and_embeddings/dual_attention_59_59_slim.pth"
PROT_PARQUET = "models_and_embeddings/protein_embeddings_ad.parquet"
DRUG_MODEL_PATH = "models_and_embeddings/gpcr_ad_finetuned.pt"
CLINICAL_MODEL_PATH = "models_and_embeddings/Decision_Tree_model.pkl"

# --- Initialization ---
mri_engine = AlzheimerInferenceEngine(MRI_MODEL_PATH) if os.path.exists(MRI_MODEL_PATH) else None
drug_engine = DrugRepurposingEngine(DRUG_MODEL_PATH, PROT_PARQUET) if os.path.exists(DRUG_MODEL_PATH) else None
clinical_engine = ClinicalAlzheimerInference(CLINICAL_MODEL_PATH) if os.path.exists(CLINICAL_MODEL_PATH) else None

class PatientData(BaseModel):
    Age: int; Gender: int; Ethnicity: int; EducationLevel: int; BMI: float
    Smoking: int; AlcoholConsumption: float; PhysicalActivity: float
    DietQuality: float; SleepQuality: float; FamilyHistoryAlzheimers: int
    CardiovascularDisease: int; Diabetes: int; Depression: int
    HeadInjury: int; Hypertension: int; SystolicBP: float; DiastolicBP: float
    CholesterolTotal: float; CholesterolLDL: float; CholesterolHDL: float
    CholesterolTriglycerides: float; MMSE: float; FunctionalAssessment: float
    ADL: float; MemoryComplaints: int; BehavioralProblems: int
    Confusion: int; Disorientation: int; PersonalityChanges: int
    DifficultyCompletingTasks: int; Forgetfulness: int

# --- API Routes (must be defined BEFORE static file mounting) ---

@app.get("/api/example-patient")
async def get_example_patient():
    """Returns dummy data for the 'Load Example' button"""
    return {
        "Age": 72, "Gender": 1, "Ethnicity": 0, "EducationLevel": 2,
        "BMI": 26.5, "Smoking": 0, "AlcoholConsumption": 3.5,
        "PhysicalActivity": 4.2, "DietQuality": 6.8, "SleepQuality": 5.5,
        "FamilyHistoryAlzheimers": 1, "CardiovascularDisease": 0, "Diabetes": 1,
        "Depression": 0, "HeadInjury": 0, "Hypertension": 1,
        "SystolicBP": 138.0, "DiastolicBP": 85.0, "CholesterolTotal": 210.0,
        "CholesterolLDL": 135.0, "CholesterolHDL": 48.0, "CholesterolTriglycerides": 165.0,
        "MMSE": 22.0, "FunctionalAssessment": 4.5, "ADL": 5.2,
        "MemoryComplaints": 1, "BehavioralProblems": 1, "Confusion": 1,
        "Disorientation": 0, "PersonalityChanges": 1, "DifficultyCompletingTasks": 1,
        "Forgetfulness": 1
    }

@app.post("/api/predict-clinical")
async def predict_clinical(patient: PatientData):
    if not clinical_engine:
        raise HTTPException(status_code=503, detail="Clinical Model not found.")
    return clinical_engine.predict(patient.dict())

@app.post("/api/predict-mri")
async def predict_mri(file: UploadFile = File(...)):
    if not mri_engine: raise HTTPException(status_code=503, detail="MRI Model missing.")
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as b: shutil.copyfileobj(file.file, b)
    try: return mri_engine.predict(temp_path)
    finally: os.remove(temp_path)

@app.get("/api/repurpose")
async def get_repurpose(query: str):
    if not drug_engine: raise HTTPException(status_code=503, detail="Drug Engine missing.")
    return drug_engine.repurpose(query)

# --- Serve React Frontend ---
# Check if dist folder exists
if os.path.exists("dist"):
    # Mount static files (JS, CSS, images, etc.)
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")
    
    # Serve index.html for all other routes (React Router support)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # If requesting a specific file that exists, serve it
        file_path = os.path.join("dist", full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Otherwise, serve index.html (for React Router)
        return FileResponse("dist/index.html")
else:
    print("⚠️  Warning: 'dist' folder not found. Frontend will not be served.")
    print("   Build your React app with 'npm run build' and place the 'dist' folder here.")

if __name__ == "__main__":
    print("🚀 Starting Integrated Alzheimer's AI Suite")
    print("📊 MRI Engine:", "✅ Loaded" if mri_engine else "❌ Not Found")
    print("💊 Drug Engine:", "✅ Loaded" if drug_engine else "❌ Not Found")
    print("🩺 Clinical Engine:", "✅ Loaded" if clinical_engine else "❌ Not Found")
    print("🌐 Frontend:", "✅ Serving from /dist" if os.path.exists("dist") else "❌ Not Found")
    print("\n🎯 Application running at: http://localhost:8000")
    print("📡 API docs available at: http://localhost:8000/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)