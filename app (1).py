"""
MediScout Web Application
- Simple web interface for the MediScout system
- Uses the balanced model for more accurate predictions
"""
import os
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import uuid
import pickle
import numpy as np
from simple_model_web import extract_features
import csv
from datetime import datetime

app = Flask(__name__)

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure app to serve uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add this line to explicitly set which model file to use
os.environ['MEDISCOUT_MODEL_PATH'] = 'medi_scout_balanced_rf_model.pkl'

# Define the condition labels
LABELS = [
    "monsoon_febrile", "diarrheal_burden", "stunting_u5", "wasting_u5",
    "tb_incidence", "obstetric_comp", "hypertension", "diabetes",
    "immunization", "ari_u5"
]

# Define symptom keywords for different conditions - same as in enhanced_predict.py
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

# Add these disease categories after your CONDITION_KEYWORDS
# Map conditions to broader medical categories
CONDITION_CATEGORIES = {
    "respiratory": ["ari_u5", "tb_incidence"],
    "gastrointestinal": ["diarrheal_burden"],
    "growth_development": ["stunting_u5", "wasting_u5"],
    "vector_borne": ["monsoon_febrile"],
    "chronic_disease": ["hypertension", "diabetes"],
    "maternal_health": ["obstetric_comp"],
    "preventive_care": ["immunization"]
}

# Map categories to descriptions for UI
CATEGORY_DESCRIPTIONS = {
    "respiratory": "Conditions affecting breathing and lungs",
    "gastrointestinal": "Conditions affecting the digestive system",
    "growth_development": "Issues related to child growth and development",
    "vector_borne": "Diseases transmitted by insects or other vectors",
    "chronic_disease": "Long-term conditions requiring ongoing management",
    "maternal_health": "Conditions related to pregnancy and childbirth",
    "preventive_care": "Preventive health measures and immunizations"
}

# Define high-risk conditions and their detection criteria
HIGH_RISK_CONDITIONS = {
    "dengue_hemorrhagic_fever": {
        "symptoms": ["bleeding", "rash", "fever", "dengue", "severe headache", "pain behind eyes"],
        "min_matches": 3,
        "warning": "Suspected dengue hemorrhagic fever - URGENT REFERRAL REQUIRED"
    },
    "severe_dehydration": {
        "symptoms": ["severe dehydration", "no urination", "very dry", "sunken eyes", "lethargic", "unconscious"],
        "min_matches": 2,
        "warning": "Severe dehydration - URGENT REHYDRATION NEEDED"
    },
    "pneumonia": {
        "symptoms": ["difficulty breathing", "fast breathing", "chest indrawing", "blue lips", "unable to drink"],
        "min_matches": 2,
        "warning": "Severe pneumonia suspected - URGENT MEDICAL CARE NEEDED"
    },
    "severe_malnutrition": {
        "symptoms": ["severe wasting", "edema", "unable to eat", "very thin", "swollen feet", "skin peeling"],
        "min_matches": 2,
        "warning": "Severe acute malnutrition - IMMEDIATE NUTRITIONAL INTERVENTION NEEDED"
    }
}

# Load the balanced model
try:
    with open('medi_scout_balanced_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Balanced model loaded successfully")
except Exception as e:
    print(f"Error loading balanced model: {e}")
    # Try to load the original model as fallback
    try:
        with open('medi_scout_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Original model loaded as fallback")
    except:
        print("No model found. Predictions will not work.")
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
    """Make combined prediction using image and symptoms"""
    # First get image-based predictions
    image_predictions = {}
    
    # Extract image features
    if image_path and os.path.exists(image_path):
        try:
            features = extract_features(image_path)
            
            if np.all(features == 0):
                print("Feature extraction failed or image is invalid")
                image_valid = False
            else:
                image_valid = True
                
                if model is not None:
                    probabilities = model.predict_proba([features])[0]
                    classes = model.classes_
                    
                    for i, condition in enumerate(classes):
                        image_predictions[condition] = float(probabilities[i] * 100)
        except Exception as e:
            print(f"Error during image prediction: {e}")
            image_valid = False
    else:
        print("No valid image path provided")
        image_valid = False
    
    # Analyze symptoms
    symptom_scores = analyze_symptoms(symptoms)
    
    # Combine image and symptom predictions
    combined_scores = {}
    
    # Start with image predictions
    for condition, prob in image_predictions.items():
        combined_scores[condition] = prob
    
    # Add symptom scores
    max_symptom_score = 1  # Avoid division by zero
    if symptom_scores:
        max_symptom_score = max([info['score'] for info in symptom_scores.values()])
    
    for condition, info in symptom_scores.items():
        symptom_weight = 60.0  # How much weight to give symptoms vs. image
        if condition in combined_scores:
            combined_scores[condition] += (info['score'] / max_symptom_score) * symptom_weight
        else:
            combined_scores[condition] = (info['score'] / max_symptom_score) * symptom_weight
    
    # Add keyword specific boosts for diarrheal conditions
    if symptoms and ("diarrhea" in symptoms.lower() or "loose stool" in symptoms.lower()):
        if "diarrheal_burden" in combined_scores:
            combined_scores["diarrheal_burden"] += 40.0
        else:
            combined_scores["diarrheal_burden"] = 40.0
    
    # Consider age for pediatric or elderly conditions
    age = patient_info.get('age')
    if age is not None:
        # For children under 5
        if age < 5:
            for condition in ["stunting_u5", "wasting_u5", "ari_u5", "diarrheal_burden"]:
                if condition in combined_scores:
                    combined_scores[condition] *= 1.5
        # For elderly
        elif age > 65:
            for condition in ["hypertension", "diabetes"]:
                if condition in combined_scores:
                    combined_scores[condition] *= 1.3
    
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
    
    # Determine disease category
    disease_category = "unknown"
    for category, conditions in CONDITION_CATEGORIES.items():
        if prediction in conditions:
            disease_category = category
            break
    
    # Check for high-risk emergency conditions
    emergency_warnings = []
    symptoms_lower = symptoms.lower() if symptoms else ""
    
    for condition, criteria in HIGH_RISK_CONDITIONS.items():
        matches = 0
        matched_symptoms = []
        
        for symptom in criteria["symptoms"]:
            if symptom in symptoms_lower:
                matches += 1
                matched_symptoms.append(symptom)
        
        if matches >= criteria["min_matches"]:
            emergency_warnings.append({
                "condition": condition,
                "warning": criteria["warning"],
                "matched_symptoms": matched_symptoms
            })
    
    # If we have emergency warnings, set risk to high
    if emergency_warnings:
        risk_level = "High"
        for warning in emergency_warnings:
            risk_factors.append(warning["warning"])
    
    # Calculate risk percentage based on various factors
    risk_percentage = 0
    
    # Base risk from prediction confidence
    if confidence > 0.8:
        risk_percentage += 30
    elif confidence > 0.6:
        risk_percentage += 20
    elif confidence > 0.4:
        risk_percentage += 10
    
    # Risk from age factors
    age = patient_info.get('age')
    if age is not None:
        if age < 5 or age > 65:  # High risk age groups
            risk_percentage += 15
        elif age < 12 or age > 55:  # Moderate risk age groups
            risk_percentage += 10
    
    # Risk from emergency conditions
    if emergency_warnings:
        risk_percentage += 30 * len(emergency_warnings)  # Each emergency adds 30%
    
    # Risk from high confidence prediction
    if top_predictions and top_predictions[0]['probability'] > 75:
        risk_percentage += 15
    
    # Cap at 100%
    risk_percentage = min(risk_percentage, 100)
    
    # Determine risk category
    if risk_percentage >= 75:
        risk_category = "Critical"
    elif risk_percentage >= 50:
        risk_category = "High" 
    elif risk_percentage >= 25:
        risk_category = "Medium"
    else:
        risk_category = "Low"
    
    # Always set high_risk flag if risk_percentage is high enough
    if risk_percentage >= 50:
        risk_level = "High"
    
    # Now return the enhanced result dictionary with risk metrics
    result = {
        "predicted_condition": prediction,
        "disease_category": disease_category,
        "category_description": CATEGORY_DESCRIPTIONS.get(disease_category, ""),
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        "top_predictions": top_predictions,
        "high_risk": risk_level == "High",
        "risk_factors": risk_factors,
        "risk_percentage": risk_percentage,  # New field
        "risk_category": risk_category,      # New field
        "emergency_warnings": emergency_warnings,
        "patient_info": patient_info,
        "symptoms": symptoms,
        "symptom_analysis": symptom_scores
    }
    
    # Save prediction to CSV
    save_prediction_to_csv(patient_info, result)
    
    return result

@app.route('/')
def index():
    return render_template('index.html', conditions=LABELS)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '')
    patient_info = {
        'age': int(request.form.get('age', 0)) if request.form.get('age') else None,
        'gender': request.form.get('gender', ''),
        'location': request.form.get('location', '')
    }
    
    # Handle image upload
    image_path = None
    if 'file' in request.files and request.files['file'].filename:
        image = request.files['file']
        filename = str(uuid.uuid4()) + os.path.splitext(image.filename)[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        # Create URL for the image in templates
        image_url = f'/uploads/{filename}'
    else:
        image_url = None
    
    # Create input data dict for logging
    input_data = {
        'symptoms': symptoms,
        'patient_info': patient_info,
        'image_path': image_path
    }
    
    # Make prediction
    result = predict_combined(image_path, symptoms, patient_info)
    
    # Save the prediction data to CSV
    save_prediction_to_csv(input_data, result)
    
    return render_template(
        'result.html',
        result=result,
        image_path=image_url,  # Use URL for HTML display
        prediction=result['predicted_condition'],
        top_predictions=result['top_predictions'],
        patient_info=patient_info,
        symptoms=symptoms
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Get JSON data
    data = request.json
    symptoms = data.get('symptoms', '')
    patient_info = data.get('patient_info', {})
    
    # Image must be uploaded separately
    image_path = None
    if 'image_path' in data:
        image_path = data['image_path']
    
    # Create input data dict for logging
    input_data = {
        'symptoms': symptoms,
        'patient_info': patient_info,
        'image_path': image_path
    }
    
    # Make prediction
    result = predict_combined(image_path, symptoms, patient_info)
    
    # Save the prediction data to CSV
    save_prediction_to_csv(input_data, result)
    
    # Add category information to API response
    api_response = result.copy()
    
    # Add emergency triage status
    api_response['requires_urgent_care'] = len(result['emergency_warnings']) > 0
    
    return jsonify(api_response)

def save_prediction_to_csv(input_data, result):
    """Save prediction inputs and outputs to a CSV file for tracking/audit"""
    csv_file = 'medi_scout_output.csv'
    file_exists = os.path.isfile(csv_file)
    
    # Define the fields to include
    fieldnames = [
        'timestamp',
        'patient_age',
        'patient_gender', 
        'patient_location',
        'symptoms_text',
        'has_image',
        'image_path',
        'predicted_condition',
        'disease_category',
        'confidence',
        'risk_percentage',
        'risk_category',
        'high_risk',
        'emergency_warnings',
        'top_predictions'
    ]
    
    # Prepare the data to write
    row_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_age': input_data.get('patient_info', {}).get('age', ''),
        'patient_gender': input_data.get('patient_info', {}).get('gender', ''),
        'patient_location': input_data.get('patient_info', {}).get('location', ''),
        'symptoms_text': input_data.get('symptoms', ''),
        'has_image': bool(input_data.get('image_path')),
        'image_path': input_data.get('image_path', ''),
        'predicted_condition': result.get('predicted_condition', ''),
        'disease_category': result.get('disease_category', ''),
        'confidence': result.get('confidence', 0),
        'risk_percentage': result.get('risk_percentage', 0),
        'risk_category': result.get('risk_category', ''),
        'high_risk': result.get('high_risk', False),
        'emergency_warnings': len(result.get('emergency_warnings', [])),
        'top_predictions': str(result.get('top_predictions', []))
    }
    
    # Write to the CSV file
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write headers if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(row_data)
    
    print(f"Prediction saved to {csv_file}")

if __name__ == '__main__':
    app.run(debug=True) 