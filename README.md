# Advanced Placement Prediction Engine

An advanced Machine Learning career prediction engine that forecasts student placement probability. It utilizes Scikit-Learn preprocessing pipelines, text-based feature extraction (NLP), and a premium glassmorphic Flask web interface.

## 📌 Upgraded Features

* **Natural Language Feature Extraction**: Accepts actual technical skills (e.g., `Python, SQL, React, AWS`) and certificate/internship names (e.g., `Google Developer Intern`, `AWS Cloud Practitioner`) instead of primitive binary flags.
* **Robust ML Pipeline**: Implements Scikit-Learn `Pipeline` and `ColumnTransformer` to handle both numeric scales (`CGPA` via `StandardScaler`) and multi-label text tokenization (`Skills` & `Certificates` via custom-tokenized `TfidfVectorizer`).
* **Model Serialization**: Automatically compares classifiers (Logistic Regression, Random Forest), selects the best-performing model using 5-fold cross-validation, and serializes the complete preprocessing + inference pipeline to `model.pkl`.
* **Premium Web Dashboard**: Beautiful, dark-themed glassmorphism interface built with Vanilla HTML/CSS. Includes dynamic circular probability meters, input suggestions, and responsive controls.
* **Interactive CLI Fallback**: Standard command-line script supports training, validation metrics output, and interactive terminal prediction.

## 🛠️ Technologies Used

* Python 3
* Scikit-Learn
* Pandas & NumPy
* Flask
* Joblib
* Vanilla HTML5 & CSS3 (Glassmorphism & HSL design system)

## 📁 Dataset & Preprocessing

The project utilizes `placement.csv` containing:
1. **CGPA**: Academic performance score (continuous scale).
2. **Skills**: Comma-separated technical skills.
3. **Certificates**: Comma-separated certifications and internship titles.
4. **Placed**: Placement outcome (binary target).

## 🚀 How to Run

### 1. Generate/Train & Predict (CLI)
To run the ML training pipeline, generate validation metrics, and test via the terminal CLI:
```bash
python placement_prediction.py
```

### 2. Run the Web Application
To start the Flask-based career portal dashboard:
```bash
python app.py
```
Then navigate to `http://127.0.0.1:5000` in your web browser.

---

## Author
Ramapriya
