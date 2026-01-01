import joblib  # Switched from pickle
import numpy as np
import pandas as pd
import os

class ClinicalAlzheimerInference:
    def __init__(self, model_path, scaler_path=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Use joblib to load the model
        self.model = joblib.load(model_path)
        
        # Define feature columns in exact order
        self.feature_columns = [
            'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 
            'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 
            'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers', 
            'CardiovascularDisease', 'Diabetes', 'Depression', 
            'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 
            'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
            'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 
            'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion', 
            'Disorientation', 'PersonalityChanges', 
            'DifficultyCompletingTasks', 'Forgetfulness'
        ]
        
        self.scale_columns = [
            'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 
            'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 
            'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 
            'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
        ]

    def preprocess_input(self, patient_data):
        df = pd.DataFrame([patient_data])
        for col in self.feature_columns:
            if col not in df.columns: df[col] = 0
        df = df[self.feature_columns]

        # Manual Scaling/Normalization
        for col in self.scale_columns:
            if col in df.columns:
                if col == 'Age': df[col] = (df[col] - 40) / 60
                elif col == 'BMI': df[col] = (df[col] - 15) / 35
                elif col == 'MMSE': df[col] = df[col] / 30
                else: df[col] = df[col] / (df[col].max() if df[col].max() > 0 else 1)
        return df.values

    def predict(self, patient_data):
        try:
            X = self.preprocess_input(patient_data)
            prediction = self.model.predict(X)[0]
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 1.0
            
            diagnosis = "Positive" if prediction == 1 else "Negative"
            return {
                "diagnosis": diagnosis,
                "prediction": int(prediction),
                "confidence": round(confidence * 100, 2),
                "risk_level": self._calculate_risk_level(patient_data),
                "message": f"Alzheimer's diagnosis: {diagnosis}"
            }
        except Exception as e:
            return {"error": str(e), "diagnosis": "Error"}

    def _calculate_risk_level(self, patient_data):
        risk_score = 0
        if patient_data.get('Age', 0) > 75: risk_score += 2
        if patient_data.get('MMSE', 30) < 24: risk_score += 3
        if patient_data.get('MemoryComplaints', 0) == 1: risk_score += 2
        return "High" if risk_score >= 5 else "Moderate" if risk_score >= 2 else "Low"

# UPDATED TRAINING HELPER
def train_and_save_model(X_train, y_train, model_save_path='Decision_Tree_model.pkl'):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=7)
    model.fit(X_train, y_train)
    # Use joblib to save
    joblib.dump(model, model_save_path)
    print(f"✓ Model saved to {model_save_path} using Joblib")