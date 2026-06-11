from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load your AI (Make sure these exact file names are inside your mysite folder!)
# We use absolute paths to ensure PythonAnywhere finds them
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'weather_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'weather_scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    # This looks for index.html inside your mysite/templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get numbers from the frontend user
    data = request.json

    # Put them in a dataframe and scale them
    df = pd.DataFrame([data])
    scaled_data = scaler.transform(df)

    # Predict (1 = rain, 0 = no rain)
    prediction = model.predict(scaled_data)[0]

    # Send result back to frontend
    result = 'rain' if prediction == 1 else 'sunny'
    return jsonify({'weather': result})