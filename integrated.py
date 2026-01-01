from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import shutil
import os
import uvicorn

# Import both engines
from mri_infer import AlzheimerInferenceEngine
from drug import DrugRepurposingEngine

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MRI Inference Engine
MRI_MODEL_PATH = "dual_attention_59_59.pth"
if not os.path.exists(MRI_MODEL_PATH):
    print(f"CRITICAL: MRI Model file {MRI_MODEL_PATH} not found!")

mri_engine = AlzheimerInferenceEngine(MRI_MODEL_PATH)

# Initialize Drug Repurposing Engine
PROT_PARQUET = "protein_embeddings_ad.parquet"
DRUG_MODEL_PATH = "gpcr_ad_finetuned.pt"

if not os.path.exists(DRUG_MODEL_PATH):
    print(f"CRITICAL: Drug Model file {DRUG_MODEL_PATH} not found!")
if not os.path.exists(PROT_PARQUET):
    print(f"CRITICAL: Protein embeddings file {PROT_PARQUET} not found!")

drug_engine = DrugRepurposingEngine(DRUG_MODEL_PATH, PROT_PARQUET)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found in current directory.</h1>"

@app.post("/predict-mri")
async def predict_mri(file: UploadFile = File(...)):
    """
    Endpoint for MRI image prediction
    """
    temp_path = f"processing_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = mri_engine.predict(temp_path)
        return result
    except Exception as e:
        print(f"MRI Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/repurpose")
async def get_repurpose(query: str):
    """
    Endpoint for drug repurposing recommendations
    """
    try:
        result = drug_engine.repurpose(query)
        return result
    except Exception as e:
        print(f"Drug Repurposing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify both engines are loaded
    """
    return {
        "status": "healthy",
        "mri_model_loaded": os.path.exists(MRI_MODEL_PATH),
        "drug_model_loaded": os.path.exists(DRUG_MODEL_PATH),
        "protein_data_loaded": os.path.exists(PROT_PARQUET)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)