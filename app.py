from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# 1. Setup absolute paths so PythonAnywhere can always find your models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'weather_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'weather_scaler.pkl')

# 2. Load the trained AI model and the data scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 3. The Home Page (Frontend)
@app.route('/')
def home():
    # This looks for index.html inside your mysite/templates folder
    return render_template('index.html') 

# 4. The AI Prediction API (Backend)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the numbers from the frontend user
        data = request.json
        
        # Put them in a pandas dataframe and scale them exactly like we did in Colab
        df = pd.DataFrame([data])
        scaled_data = scaler.transform(df)
        
        # Predict the final answer (1 = rain, 0 = no rain)
        prediction = model.predict(scaled_data)[0]
        
        # Get the exact percentage probability of Rain
        # predict_proba returns a list: [probability_of_sunny, probability_of_rain]
        probabilities = model.predict_proba(scaled_data)[0]
        rain_percentage = round(probabilities[1] * 100, 1) # Grab the rain prob and convert to %
        
        # Send result AND probability back to frontend
        result = 'rain' if prediction == 1 else 'sunny'
        
        return jsonify({
            'weather': result, 
            'probability': rain_percentage
        })
        
    except Exception as e:
        # If anything goes wrong, catch the error so the server doesn't crash
        return jsonify({'error': str(e)}), 500
