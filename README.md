# 🩺 Diabetes Prediction Web App

A Flask-based machine learning web application to predict the likelihood of diabetes using clinical parameters from the **100,000+ Diabetes Clinical Dataset** on Kaggle.

---

## 📊 Dataset

**Source:** [Kaggle - Diabetes Clinical Dataset (100,000+ records)](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset)

**Features:**
- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `smoking_history`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`
- `diabetes` (Target Variable: 0 = Negative, 1 = Positive)

---

## 🧠 Machine Learning Model

- **Model**: `RandomForestClassifier` (or any model of your choice)
- **Input Features Used**:
  - HbA1c Level
  - Blood Glucose Level
  - BMI
  - Age
- **Output**: Diabetes Prediction (Positive/Negative)
- **Model File**: `diabetes_model.pkl`

---

## 💻 Web App Features

- Input form for:
  - HbA1c Level
  - Blood Glucose Level
  - BMI
  - Age
- Real-time prediction via the trained ML model
- Displays result: **"Diabetes Positive"** or **"Diabetes Negative"**

---

## 📁 Project Structure

```

diabetes-prediction-app/
├── diabetes\_modelApp.py          # Main Flask app
├── diabetes\_model.pkl            # Trained ML model
├── requirements.txt              # Python dependencies
├── templates/
│   ├── index.html                # Input form
│   └── result.html               # Output result
└── static/                       # (optional) CSS/JS files

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/kartik10sharma/diabetes-prediction-app.git
cd diabetes-prediction-app
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and Save the Model (if not already saved)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Preprocess and train
X = df[["HbA1c_level", "blood_glucose_level", "bmi", "age"]]
y = df["diabetes"]

model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "diabetes_model.pkl")
```

### 4. Run the Flask App

```bash
python diabetes_modelApp.py
```

### 5. Open in Browser

Go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## ✅ Requirements

List of Python libraries:

```txt
Flask
pandas
numpy
scikit-learn
joblib
```

Install all using:

```bash
pip install Flask pandas numpy scikit-learn joblib
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 🙋‍♂️ Author

**Kartik Sharma**
*Artificial Intelligence and Machine Learning Enthusiast*
[GitHub](https://github.com/kartik10sharma)

---

## 📄 License

This project is for educational purposes only.
Dataset © [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset)

---


