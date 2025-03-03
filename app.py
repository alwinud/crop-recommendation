from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Importing model
import os
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Getting input values
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosporus'])
        K = int(request.form['Potassium'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        ph = float(request.form['ph'])
        Rainfall = float(request.form['Rainfall'])

        # Creating input array
        feature_list = [N, P, K, Temperature, Humidity, ph, Rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Making prediction
        prediction = model.predict(single_pred)

        # Crop dictionary (inverted for lookup)
        crop_dict = {
            'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4,
            'pigeonpeas': 5, 'mothbeans': 6, 'mungbean': 7, 'blackgram': 8,
            'lentil': 9, 'pomegranate': 10, 'banana': 11, 'mango': 12,
            'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 'apple': 16,
            'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
            'jute': 21, 'coffee': 22
        }
        
        # Reverse the dictionary for prediction lookup
        crop_lookup = {v: k for k, v in crop_dict.items()}

        # Get crop name from prediction
        crop = crop_lookup.get(prediction[0], "Unknown Crop")
        result = f"{crop} is the best crop to be cultivated right here."

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

# Running the Flask app
if __name__ == "__main__":
    app.run(debug=True)

