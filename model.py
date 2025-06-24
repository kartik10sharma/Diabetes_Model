import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
# Load dataset
data = pd.read_csv('diabetes_dataset.csv')
# Features and labels
X = data[['hbA1c_level', 'blood_glucose_level', 'bmi', 'age']]
y = data['diabetes']
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Preprocessing and model pipeline
model = Pipeline(steps=[
("scaler", StandardScaler()), # Standardize features
("classifier", LogisticRegression(max_iter=1000)) # Logistic Regression model
])
# Model training
model.fit(X_train, y_train)
# Save model
joblib.dump(model, 'diabetes_model.pkl')
print("Model saved as diabetes_model.pkl")