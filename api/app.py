from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Absolute path from current file (app.py) to model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_diabetes = pickle.load(open(os.path.join(BASE_DIR, "../diabetes/model_diabetes.pkl"), "rb"))
model_heart = pickle.load(open(os.path.join(BASE_DIR, "../heart/model_heart.pkl"), "rb"))
model_stroke = pickle.load(open(os.path.join(BASE_DIR, "../stroke/model_stroke.pkl"), "rb"))

@app.route('/')
def index():
    return "🧠 Personal Health Risk Prediction API"

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.get_json()
    features = np.array([
        data["Pregnancies"], data["Glucose"], data["BloodPressure"],
        data["SkinThickness"], data["Insulin"], data["BMI"],
        data["DiabetesPedigreeFunction"], data["Age"]
    ]).reshape(1, -1)
    prediction = model_diabetes.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    data = request.get_json()
    features = np.array([
        data["Age"], data["Sex"], data["ChestPainType"], data["RestingBP"],
        data["Cholesterol"], data["FastingBS"], data["RestingECG"],
        data["MaxHR"], data["ExerciseAngina"], data["Oldpeak"], data["ST_Slope"]
    ]).reshape(1, -1)
    prediction = model_heart.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

@app.route('/predict/stroke', methods=['POST'])
def predict_stroke():
    data = request.get_json()
    features = np.array([
        data["gender"], data["age"], data["hypertension"], data["heart_disease"],
        data["ever_married"], data["work_type"], data["Residence_type"],
        data["avg_glucose_level"], data["bmi"], data["smoking_status"]
    ]).reshape(1, -1)
    prediction = model_stroke.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
