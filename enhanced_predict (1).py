"""
Enhanced MediScout Prediction System 
- Combines image analysis with symptom data
- Provides comprehensive diagnosis and risk assessment
"""
import pickle
import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
from simple_model_web import extract_features

# Define symptom keywords for different conditions
CONDITION_KEYWORDS = {
    "monsoon_febrile": ["fever", "chills", "headache", "body ache", "mosquito", "malaria", "dengue"],
    "diarrheal_burden": ["diarrhea", "loose stool", "watery stool", "dehydration", "vomiting"],
    "stunting_u5": ["short stature", "growth", "poor growth", "malnutrition"],
    "wasting_u5": ["weight loss", "thin", "malnourished", "malnutrition"],
    "tb_incidence": ["cough", "tuberculosis", "tb", "night sweat", "weight loss", "blood sputum"],
    "obstetric_comp": ["pregnancy", "pregnant", "labor", "birth", "vaginal bleeding", "contraction"],
    "hypertension": ["high blood pressure", "headache", "dizziness", "hypertension"],
    "diabetes": ["diabetes", "thirst", "urination", "frequent urination", "sugar", "glucose"],
    "immunization": ["vaccine", "vaccination", "immunization", "shot", "booster"],
    "ari_u5": ["respiratory", "breathing", "pneumonia", "bronchitis", "breath", "cough", "cold"]
}

# Load model once at module level
try:
    with open('medi_scout_balanced_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def analyze_symptoms(symptoms_text):
    """Analyze symptoms text to find condition matches"""
    if not symptoms_text:
        return {}
    
    symptoms_text = symptoms_text.lower()
    scores = {}
    
    for condition, keywords in CONDITION_KEYWORDS.items():
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword in symptoms_text:
                score += 1
                matched_keywords.append(keyword)
        
        if matched_keywords:
            scores[condition] = {
                'score': score,
                'matched_keywords': matched_keywords
            }
    
    return scores

def predict_combined(image_path, symptoms, patient_info):
    # First get image-based predictions
    image_predictions = {}
    
    if image_path and os.path.exists(image_path):
        try:
            features = extract_features(image_path)
            print("Extracted features:", features[:5], "...")  # Print just first 5 features
            
            if np.all(features == 0):
                print("Feature extraction failed or image is invalid")
                image_valid = False
            else:
                image_valid = True
                
                if model is not None:
                    probabilities = model.predict_proba([features])[0]
                    print("Model probabilities:", probabilities)
                    classes = model.classes_
                    
                    # Apply TB correction - reduce TB bias
                    tb_index = np.where(classes == "tb_incidence")[0][0] if "tb_incidence" in classes else -1
                    if tb_index >= 0:
                        # Reduce TB probability by 50%
                        probabilities[tb_index] *= 0.5
                        # Normalize probabilities to sum to 1 again
                        probabilities = probabilities / probabilities.sum()
                    
                    for i, condition in enumerate(classes):
                        image_predictions[condition] = float(probabilities[i] * 100)
        except Exception as e:
            print(f"Error during image prediction: {e}")
            image_valid = False
    else:
        print("No valid image path provided")
        image_valid = False
    
    # Now analyze symptoms
    symptom_scores = analyze_symptoms(symptoms)
    print("Symptom analysis:", symptom_scores)
    
    # Combine image and symptom predictions
    combined_scores = {}
    
    # Start with image predictions
    for condition, prob in image_predictions.items():
        combined_scores[condition] = prob
    
    # Add symptom scores - weight symptoms MUCH more heavily
    max_symptom_score = 1  # Avoid division by zero
    if symptom_scores:
        max_symptom_score = max([info['score'] for info in symptom_scores.values()])
    
    for condition, info in symptom_scores.items():
        # Dramatically increase symptom weight to 80 (from 30)
        symptom_weight = 80.0  # How much weight to give symptoms vs. image
        if condition in combined_scores:
            combined_scores[condition] += (info['score'] / max_symptom_score) * symptom_weight
        else:
            combined_scores[condition] = (info['score'] / max_symptom_score) * symptom_weight
    
    # Add keyword specific boosts for diarrheal conditions
    if symptoms and ("diarrhea" in symptoms.lower() or "loose stool" in symptoms.lower() or 
                    "watery stool" in symptoms.lower() or "dehydration" in symptoms.lower()):
        if "diarrheal_burden" in combined_scores:
            combined_scores["diarrheal_burden"] += 50.0  # Strong boost
        else:
            combined_scores["diarrheal_burden"] = 50.0
    
    # Consider age for pediatric or elderly conditions
    age = patient_info.get('age')
    if age is not None:
        # For children under 5
        if age < 5:
            for condition in ["stunting_u5", "wasting_u5", "ari_u5", "diarrheal_burden"]:
                if condition in combined_scores:
                    combined_scores[condition] *= 1.8  # Boost these scores even more
        # For elderly
        elif age > 65:
            for condition in ["hypertension", "diabetes"]:
                if condition in combined_scores:
                    combined_scores[condition] *= 1.5  # Boost these scores
    
    # Apply TB correction factor - needed because model seems biased toward TB
    if "tb_incidence" in combined_scores:
        # Only keep 30% of the original TB score unless symptoms strongly indicate TB
        has_tb_symptoms = False
        if "tb_incidence" in symptom_scores and symptom_scores["tb_incidence"]["score"] >= 2:
            has_tb_symptoms = True
            
        if not has_tb_symptoms:
            combined_scores["tb_incidence"] *= 0.3  # Significant reduction
    
    # Get top predictions
    if not combined_scores:
        # Fallback if we have no predictions
        top_predictions = []
        prediction = "Insufficient data"
        confidence = 0.0
    else:
        sorted_conditions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [
            {"condition": cond, "probability": score} 
            for cond, score in sorted_conditions[:3]
        ]
        prediction = top_predictions[0]["condition"]
        confidence = top_predictions[0]["probability"] / 100
    
    # Risk assessment
    risk_level = "Low"
    risk_factors = []
    
    # Check high probability conditions
    if top_predictions and top_predictions[0]['probability'] > 75:
        risk_level = "High"
        risk_factors.append(f"High probability of {top_predictions[0]['condition']}")
    
    # Age-based risk factors
    if patient_info.get('age') is not None:
        if patient_info['age'] < 5 and any(c in prediction for c in ["stunting_u5", "wasting_u5", "ari_u5"]):
            risk_level = "High"
            risk_factors.append("Child under 5 with critical condition")
        elif patient_info['age'] > 65 and any(c in prediction for c in ["hypertension", "diabetes"]):
            risk_level = "High"
            risk_factors.append("Elderly patient with chronic condition")
    
    return {
        "predicted_condition": prediction,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        "top_predictions": top_predictions,
        "high_risk": risk_level == "High",
        "risk_factors": risk_factors,
        "patient_info": patient_info,
        "symptoms": symptoms,
        "symptom_analysis": symptom_scores
    }

if __name__ == "__main__":
    # Example usage from command line
    if len(sys.argv) < 2:
        print("Usage: python enhanced_predict.py <image_path> [symptoms_text]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    symptoms_text = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Analyzing image: {image_path}")
    if symptoms_text:
        print(f"Symptoms: {symptoms_text}")
    
    # Example patient info
    patient_info = {
        'age': 35,
        'gender': 'female',
        'location': 'Punjab'
    }
    
    result = predict_combined(image_path, symptoms_text, patient_info)
    
    # Display results
    print("\n===== DIAGNOSIS RESULTS =====")
    print(f"Predicted condition: {result['predicted_condition']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if result['high_risk']:
        print("\n⚠️ HIGH RISK CASE - URGENT REFERRAL RECOMMENDED ⚠️")
        print("Risk factors:")
        for factor in result['risk_factors']:
            print(f"  - {factor}")
    
    print("\nTop predictions:")
    for pred in result['top_predictions']:
        print(f"  {pred['condition']}: {pred['probability']:.2f}%")
    
    if 'symptom_analysis' in result and result['symptom_analysis']:
        print("\nSymptom analysis:")
        for condition, info in result['symptom_analysis'].items():
            print(f"  {condition}: {info['score']} matched keywords ({', '.join(info['matched_keywords'])})")