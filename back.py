from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import shutil
import os
import uvicorn

# Ensure the file is named mri_infer.py
from mri_infer import AlzheimerInferenceEngine 

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
MODEL_PATH = "dual_attention_59_59.pth"
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL: Model file {MODEL_PATH} not found!")

engine = AlzheimerInferenceEngine(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found in current directory.</h1>"

@app.post("/predict-mri")
async def predict_mri(file: UploadFile = File(...)):
    # Create a unique temp name to prevent collisions
    temp_path = f"processing_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run Inference
        result = engine.predict(temp_path)
        return result
    except Exception as e:
        # Ensure we return a 500 if things go wrong during processing
        print(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always cleanup the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Host 127.0.0.1 is standard for local dev
    uvicorn.run(app, host="127.0.0.1", port=8000)