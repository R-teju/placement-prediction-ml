from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

# Import tokenize_comma_separated so joblib can deserialize the model pipeline correctly
from placement_prediction import tokenize_comma_separated

app = Flask(__name__)

# Load model pipeline
MODEL_PATH = "model.pkl"
model_pipeline = None

def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("WARNING: model.pkl not found. Train the model first.")

# Initialize model
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model_pipeline
    if model_pipeline is None:
        # Retry loading model if it wasn't loaded before
        load_model()
        if model_pipeline is None:
            return jsonify({'error': 'Model is not trained. Please train the model first.'}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        cgpa_str = data.get('cgpa', '0')
        cgpa = float(cgpa_str) if cgpa_str else 0.0
        skills = data.get('skills', '').strip()
        certificates = data.get('certificates', '').strip()
        
        # Create input DataFrame with columns matching training data
        input_data = pd.DataFrame([{
            'CGPA': cgpa,
            'Skills': skills,
            'Certificates': certificates
        }])
        
        # Make prediction
        prediction = int(model_pipeline.predict(input_data)[0])
        probabilities = model_pipeline.predict_proba(input_data)[0]
        probability_placed = float(probabilities[1])
        
        # Generate dynamic recommendations
        status = "Placed" if prediction == 1 else "Not Placed"
        
        if prediction == 1:
            message = "Excellent profile! Your skills and academic performance align perfectly with recruiter demands."
        else:
            message = "Looks like there is room for improvement. Try boosting your CGPA, adding high-value skills (e.g. AWS, Django, PostgreSQL), or gaining practical experience through internships."
            
        return jsonify({
            'placed': prediction,
            'probability': round(probability_placed * 100, 2),
            'status': status,
            'message': message
        })
        
    except ValueError as ve:
        return jsonify({'error': f'Invalid value input: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
