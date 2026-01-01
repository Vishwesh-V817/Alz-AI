import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from numpy.linalg import norm
import os

# --- ARCHITECTURE (Matches your training code) ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.3))
    def forward(self, x): return x + self.block(self.ln(x))

class BindingModel(nn.Module):
    def __init__(self, dim=1408, zdim=1024, ncls=4):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.encoder = nn.Sequential(
            nn.Linear(dim, 2048), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(2048, zdim), nn.GELU()
        )
        self.res1 = ResidualBlock(zdim)
        self.res2 = ResidualBlock(zdim)
        self.ki = nn.Linear(zdim, ncls); self.ic = nn.Linear(zdim, ncls)
        self.ec = nn.Linear(zdim, ncls); self.kd = nn.Linear(zdim, ncls)

    def forward(self, x):
        x = self.input_norm(x)
        z = self.encoder(x)
        z = self.res1(z); z = self.res2(z)
        return [self.ki(z), self.ic(z), self.ec(z), self.kd(z)]

class DrugRepurposingEngine:
    def __init__(self, model_path, prot_parquet_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. Load Multi-Task DTI Model
        self.model = BindingModel().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # 2. Load Drug Encoder (ChemBERTa)
        print("📥 Loading ChemBERTa drug encoder (SMILES Mode)...")
        self.drug_tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.drug_enc = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", use_safetensors=True).to(self.device).eval()

        # 3. Load Protein Embeddings from Parquet
        print(f"📂 Loading protein embeddings from {prot_parquet_path}...")
        if not os.path.exists(prot_parquet_path):
            raise FileNotFoundError(f"Missing {prot_parquet_path}")
            
        prot_df = pd.read_parquet(prot_parquet_path)
        # Convert embeddings to numpy arrays if they're stored as lists
        self.prot_dict = {}
        for _, row in prot_df.iterrows():
            uid = row['UniProt_ID']
            emb = row['embedding']
            # Handle both list and numpy array formats
            if isinstance(emb, (list, tuple)):
                self.prot_dict[uid] = np.array(emb, dtype=np.float32)
            else:
                self.prot_dict[uid] = np.array(emb, dtype=np.float32)

        self.target_names = {
            "P08173": "CHRM4", "P08908": "HTR1A", "P11229": "CHRM1",
            "P14416": "DRD2", "P29274": "ADORA2A", "P41597": "CCR2",
            "P43220": "GLP1R", "P61073": "CXCR4"
        }
        
        print(f"✅ Loaded {len(self.prot_dict)} protein embeddings")
        print(f"✅ Protein embedding shape: {list(self.prot_dict.values())[0].shape}")

    def _embed_drug(self, smiles):
        inputs = self.drug_tok(smiles, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.drug_enc(**inputs).last_hidden_state
            return out.mean(dim=1).cpu().numpy()[0]

    def repurpose(self, input_val):
        # 1. Strictly treat input as SMILES (clean only formatting)
        smiles = input_val.strip("() ")
        
        try:
            drug_vec = self._embed_drug(smiles)
            print(f"Drug embedding shape: {drug_vec.shape}")
        except Exception as e:
            print(f"Error embedding drug: {str(e)}")
            return {"error": f"Invalid SMILES string provided: {str(e)}"}
        
        # 2. Similarity Ranking Logic for ALL 8 targets
        sims = []
        for uid, p_vec in self.prot_dict.items():
            if uid not in self.target_names: 
                continue
            
            # Ensure p_vec is numpy array
            if not isinstance(p_vec, np.ndarray):
                p_vec = np.array(p_vec, dtype=np.float32)
            
            # Compare drug feature subset to protein feature
            # Adjust slicing based on actual protein embedding size
            prot_size = p_vec.shape[0]
            d_subset = drug_vec[:prot_size]  # Match protein embedding size
            
            # Calculate cosine similarity
            d_norm = norm(d_subset)
            p_norm = norm(p_vec)
            
            if d_norm > 0 and p_norm > 0:
                s = np.dot(d_subset, p_vec) / (d_norm * p_norm)
            else:
                s = 0.0
            
            sims.append({
                "uid": uid, 
                "name": self.target_names[uid], 
                "score": float(s)
            })

        # 3. Sort all 8 targets by similarity (keep all of them)
        all_targets = sorted(sims, key=lambda x: x["score"], reverse=True)
        print(f"All {len(all_targets)} targets: {[(t['name'], t['score']) for t in all_targets]}")
        
        # 4. Predict Binding for ALL targets
        results = []
        class_map = {0: "Weak", 1: "Moderate", 2: "Strong", 3: "Elite"}
        
        for target in all_targets:
            p_vec = self.prot_dict[target["uid"]]
            
            # Ensure proper dimensions for concatenation
            drug_tensor = torch.tensor(drug_vec, dtype=torch.float32)
            prot_tensor = torch.tensor(p_vec, dtype=torch.float32)
            
            # Concatenate to match model input dimension (1408)
            # If dimensions don't match, pad or truncate
            expected_dim = 1408
            total_dim = drug_tensor.shape[0] + prot_tensor.shape[0]
            
            if total_dim < expected_dim:
                # Pad with zeros
                x = torch.cat([drug_tensor, prot_tensor], dim=0)
                padding = torch.zeros(expected_dim - total_dim)
                x = torch.cat([x, padding], dim=0)
            elif total_dim > expected_dim:
                # Truncate drug embedding to fit
                drug_size = expected_dim - prot_tensor.shape[0]
                x = torch.cat([drug_tensor[:drug_size], prot_tensor], dim=0)
            else:
                x = torch.cat([drug_tensor, prot_tensor], dim=0)
            
            x = x.to(self.device)
            
            with torch.no_grad():
                outs = self.model(x.unsqueeze(0))
                # Apply softmax to get confidence percentages
                probs = [F.softmax(head_logits, dim=1) for head_logits in outs]
            
            preds = {}
            for i, head in enumerate(["Ki", "IC50", "EC50", "Kd"]):
                # Get the max value (confidence) and the index (the class)
                conf, cls_idx = torch.max(probs[i], dim=1)
                
                preds[head] = {
                    "label": class_map[cls_idx.item()],
                    "confidence": round(float(conf.item()) * 100, 2)
                }
            
            results.append({
                "target": target["name"], 
                "uniprot": target["uid"], 
                "similarity": round(target["score"] * 100, 2),
                "predictions": preds
            })

        print(f"✅ Returning {len(results)} results")
        return {"smiles": smiles, "results": results}