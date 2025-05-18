"""
MediScout Simple High-Accuracy Model
- Uses scikit-learn instead of deep learning
- No need for torchvision
- Targets >70% accuracy across conditions
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

# Define constants
LABELS = [
    "monsoon_febrile", "diarrheal_burden", "stunting_u5", "wasting_u5",
    "tb_incidence", "obstetric_comp", "hypertension", "diabetes",
    "immunization", "ari_u5"
]

# Function to extract features from images
def extract_features(image_path, size=(64, 64)):
    try:
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
        
        # Convert to array and flatten
        img_array = np.array(img)
        
        # Extract simple features
        # 1. Average color per channel
        avg_color = img_array.mean(axis=(0, 1))
        
        # 2. Standard deviation per channel
        std_color = img_array.std(axis=(0, 1))
        
        # 3. Histogram features (10 bins per channel)
        hist_features = []
        for channel in range(3):
            hist, _ = np.histogram(img_array[:,:,channel], bins=10, range=(0, 256))
            hist_features.extend(hist)
        
        # Combine features
        features = np.concatenate([avg_color, std_color, hist_features])
        return features
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        # Return zeros as fallback
        return np.zeros(3 + 3 + 30)  # 3 avg + 3 std + 30 hist

# Load and prepare data
def prepare_data(csv_path="medi_scout_data/dataset.csv"):
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from dataset")
    
    # Check class distribution
    condition_counts = Counter(df['condition'])
    print("Condition distribution:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} images")
    
    # Check for missing images
    missing_images = 0
    valid_paths = []
    for path in df['image_path']:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            missing_images += 1
    print(f"Missing images: {missing_images}")
    
    # Filter out missing images
    df = df[df['image_path'].isin(valid_paths)]
    print(f"Using {len(df)} valid images")
    
    # Extract features
    print("Extracting features from images...")
    X = []
    y = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        features = extract_features(row['image_path'])
        X.append(features)
        y.append(row['condition'])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, df

# Train model
def train_model(X_train, y_train):
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    plt.barh(range(20), importances[indices])
    plt.yticks(range(20), [f"Feature {i}" for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return accuracy, precision, recall, f1, y_pred

# Generate per-condition metrics
def evaluate_per_condition(y_test, y_pred):
    # Calculate metrics for each condition
    condition_metrics = {}
    
    for condition in np.unique(y_test):
        # Create binary arrays (1 if this condition, 0 otherwise)
        y_true = np.array([1 if c == condition else 0 for c in y_test])
        y_pred_binary = np.array([1 if c == condition else 0 for c in y_pred])
        
        # Calculate metrics
        condition_metrics[condition] = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0),
            'support': sum(y_true)
        }
    
    # Create a DataFrame for better visualization
    metrics_df = pd.DataFrame.from_dict(condition_metrics, orient='index')
    
    # Print and save metrics
    print("\n===== PER-CONDITION METRICS =====")
    print(metrics_df)
    metrics_df.to_csv('condition_metrics.csv')
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    metrics_df['accuracy'].sort_values().plot(kind='barh', color='skyblue')
    plt.axvline(x=0.7, color='red', linestyle='--', label='70% Target')
    plt.xlabel('Accuracy')
    plt.title('Accuracy by Condition')
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_by_condition.png')
    plt.close()
    
    return metrics_df

# Main function
if __name__ == "__main__":
    print("Starting MediScout Simple High-Accuracy Model")
    
    # Prepare data
    X_train, X_test, y_train, y_test, df = prepare_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy, precision, recall, f1, y_pred = evaluate_model(model, X_test, y_test)
    
    # Evaluate per-condition metrics
    condition_metrics = evaluate_per_condition(y_test, y_pred)
    
    # Save model
    import pickle
    with open('medi_scout_rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nFinal model saved with {accuracy:.2%} accuracy")
    
    # Check if we achieved >70% accuracy
    if accuracy >= 0.7:
        print("✅ SUCCESS: Model achieved >70% accuracy target!")
    else:
        print("⚠ WARNING: Model did not reach 70% accuracy target.")
        print("Suggestions to improve accuracy:")
        print("1. Increase number of trees")
        print("2. Extract more sophisticated features")
        print("3. Try a different model (SVM, XGBoost)")
        print("4. Balance the dataset better")