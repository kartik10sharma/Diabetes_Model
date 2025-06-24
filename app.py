from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize app
app = Flask(__name__)

# Load model
model = joblib.load("diabetes_model.pkl")

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form  # Use request.form to get form data

    # Prepare input data for prediction
    input_data = {
        "hbA1c_level": [float(data["hbA1c_level"])],
        "blood_glucose_level": [float(data["blood_glucose_level"])],
        "bmi": [float(data["bmi"])],
        "age": [int(data["age"])]
    }

    # Create DataFrame for the model
    input_df = pd.DataFrame(input_data)

    # Make prediction
    prediction = model.predict(input_df)

    # Return result
    result = "Diabetes Positive" if prediction[0] == 1 else "Diabetes Negative"
    return render_template('result.html', result=result)  # Render result.html with the prediction result

if __name__ == "__main__":
    app.run(debug=True)
