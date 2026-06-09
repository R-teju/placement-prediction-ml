import os
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Custom tokenizer for comma-separated lists of skills and certificates
def tokenize_comma_separated(text):
    if not isinstance(text, str):
        return []
    # Lowercase, split by commas, and strip whitespace
    return [item.strip().lower() for item in text.split(',') if item.strip()]

def train_and_save_model():
    # 1. Load dataset
    if not os.path.exists("placement.csv"):
        raise FileNotFoundError("placement.csv dataset not found. Please run the data generator first.")
        
    df = pd.read_csv("placement.csv")
    print("Dataset loaded successfully. Shape:", df.shape)
    
    # Fill any missing values
    df['CGPA'] = df['CGPA'].fillna(df['CGPA'].mean())
    df['Skills'] = df['Skills'].fillna('None')
    df['Certificates'] = df['Certificates'].fillna('None')
    
    # 2. Split input and output
    X = df[['CGPA', 'Skills', 'Certificates']]
    y = df['Placed']
    
    # 3. Create Pipeline Preprocessor
    # We use TfidfVectorizer with our custom tokenizer to convert comma-separated string lists to features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cgpa_scaler', StandardScaler(), ['CGPA']),
            ('skills_vectorizer', TfidfVectorizer(tokenizer=tokenize_comma_separated, token_pattern=None), 'Skills'),
            ('certs_vectorizer', TfidfVectorizer(tokenizer=tokenize_comma_separated, token_pattern=None), 'Certificates')
        ],
        remainder='drop'
    )
    
    # 4. Define candidate models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42)
    }
        
    best_model_name = None
    best_pipeline = None
    best_score = -1
    
    # Train-test split for metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Evaluate models using Cross-Validation
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)
        print(f"{name} Cross-Validation Accuracy: {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model selected: {best_model_name} (CV Accuracy: {best_score:.4f})")
    
    # Fit the best pipeline on training data
    best_pipeline.fit(X_train, y_train)
    
    # Evaluate on holdout test set
    y_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Fit on all data before saving for production use
    print("Fitting model on complete dataset...")
    best_pipeline.fit(X, y)
    
    # Save the pipeline
    joblib.dump(best_pipeline, "model.pkl")
    print("Trained model pipeline saved successfully to model.pkl\n")
    
    return best_pipeline

if __name__ == "__main__":
    # Force train/retrain to verify the pipeline runs end-to-end
    pipeline = train_and_save_model()
            
    print("========== Interactive ML Prediction CLI ==========")
    try:
        cgpa_input = input("Enter CGPA (example: 8.5): ").strip()
        cgpa = float(cgpa_input) if cgpa_input else 0.0
        
        skills = input("Enter Skills (separated by commas, e.g., Python, SQL, React): ").strip()
        certificates = input("Enter Internships/Certificates (separated by commas, e.g., AWS, Google Intern): ").strip()
        
        print("===================================================")
        
        # Predict
        input_data = pd.DataFrame([{
            'CGPA': cgpa,
            'Skills': skills,
            'Certificates': certificates
        }])
        
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        placed_prob = probabilities[1]
        
        print(f"\nPlacement Probability: {placed_prob * 100:.2f}%")
        if prediction == 1:
            print("Result: Student is predicted to be PLACED!")
        else:
            print("Result: Student is predicted to be NOT PLACED.")
            
    except ValueError as e:
        print("Invalid input value:", e)