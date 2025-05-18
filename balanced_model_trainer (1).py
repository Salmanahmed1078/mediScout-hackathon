"""
Balanced MediScout Model Trainer
- Creates a balanced dataset with equal samples per condition
- Ensures no bias toward any single condition
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
from simple_model_web import extract_features

# Define constants
LABELS = [
    "monsoon_febrile", "diarrheal_burden", "stunting_u5", "wasting_u5",
    "tb_incidence", "obstetric_comp", "hypertension", "diabetes",
    "immunization", "ari_u5"
]

def prepare_balanced_data(csv_path="medi_scout_data/dataset.csv", target_samples=20):
    """
    Prepare a balanced dataset with equal samples per condition
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images from dataset")
    
    # Check class distribution
    condition_counts = Counter(df['condition'])
    print("Original condition distribution:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} images")
    
    # Check for missing images
    df = df[df['image_path'].apply(os.path.exists)]
    print(f"Using {len(df)} valid images")
    
    # Create balanced dataset
    balanced_df = pd.DataFrame()
    
    for condition in LABELS:
        condition_df = df[df['condition'] == condition]
        
        if len(condition_df) == 0:
            print(f"WARNING: No samples for {condition}")
            continue
            
        # If we have fewer samples than target, duplicate some
        if len(condition_df) < target_samples:
            # Duplicate samples with replacement
            indices = np.random.choice(condition_df.index, target_samples - len(condition_df), replace=True)
            extra_samples = df.loc[indices]
            condition_df = pd.concat([condition_df, extra_samples])
        
        # If we have more samples than target, take a random subset
        elif len(condition_df) > target_samples:
            condition_df = condition_df.sample(target_samples, random_state=42)
        
        balanced_df = pd.concat([balanced_df, condition_df])
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify balanced distribution
    balanced_counts = Counter(balanced_df['condition'])
    print("\nBalanced condition distribution:")
    for condition, count in balanced_counts.items():
        print(f"  {condition}: {count} images")
    
    return balanced_df

def extract_features_batch(df):
    """Extract features from all images in the dataframe"""
    X = []
    y = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        features = extract_features(row['image_path'])
        X.append(features)
        y.append(row['condition'])
    
    return np.array(X), np.array(y)

def train_balanced_model():
    """Train a new model with balanced data"""
    # Prepare balanced data
    balanced_df = prepare_balanced_data()
    
    # Extract features
    print("Extracting features from balanced dataset...")
    X, y = extract_features_batch(balanced_df)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train model
    print("Training balanced Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBalanced model accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Balanced Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('balanced_confusion_matrix.png')
    
    # Save the balanced model
    with open('medi_scout_balanced_rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Balanced model saved to medi_scout_balanced_rf_model.pkl")
    
    # Modify enhanced_predict.py to use the new balanced model
    update_model_path()
    
    return model

def update_model_path():
    """Update the model path in enhanced_predict.py"""
    try:
        # Open with UTF-8 encoding
        with open('enhanced_predict.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "with open('medi_scout_rf_model.pkl', 'rb') as f:" in line:
                lines[i] = "    with open('medi_scout_balanced_rf_model.pkl', 'rb') as f:\n"
                
        # Write with UTF-8 encoding
        with open('enhanced_predict.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        print("Updated enhanced_predict.py to use the balanced model")
    except Exception as e:
        print(f"Error updating enhanced_predict.py: {e}")
        # Let's get more information about the file
        try:
            import os
            print(f"File size: {os.path.getsize('enhanced_predict.py')} bytes")
            # Read in binary mode to detect encoding
            with open('enhanced_predict.py', 'rb') as f:
                content = f.read()
                print(f"Problematic byte at position 1038: {hex(content[1038])}")
        except Exception as inner_e:
            print(f"Error analyzing file: {inner_e}")

if __name__ == "__main__":
    train_balanced_model() 