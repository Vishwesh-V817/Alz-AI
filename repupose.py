from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from drug import DrugRepurposingEngine

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Path to the parquet you generated in Colab
PROT_PARQUET = "protein_embeddings_ad.parquet"
MODEL_PATH = "gpcr_ad_finetuned.pt"

# Initialize with the new parameter
engine = DrugRepurposingEngine(MODEL_PATH, PROT_PARQUET)

@app.get("/repurpose")
async def get_repurpose(query: str):
    # This calls engine.repurpose(query)
    return engine.repurpose(query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)