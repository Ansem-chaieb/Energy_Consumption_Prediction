import os
import datetime
import logging
import traceback
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model_path = 'models/lstm_model.h5'

try:
    model = load_model(model_path)
    data = pd.read_csv('test_data.csv')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading model: %s", e)
    traceback.print_exc()

@app.route('/predict', methods=['GET'])
def predict():
      try:
            full_datetime = request.args.get('datetime')
            # Select features

            feature_columns = ['voltage', 'reactive_power', 'apparent_power', 'power_factor', 
                                    'temp', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity',
                                    'speed', 'deg', 'active_power_lag_1', 'active_power_lag_8', 'year',
                                    'month_of_year', 'week_of_year', 'day_of_year', 'day_of_week', 'hour_of_day', 
                                    'is_weekend', 'is_holiday', 'main_Clouds', 'main_Drizzle', 'main_Fog',
                                    'main_Haze', 'main_Mist', 'main_Rain', 'main_Thunderstorm', 
                                    'description_clear sky', 'description_drizzle', 'description_few clouds', 
                                    'description_fog', 'description_haze', 'description_heavy intensity rain', 
                                    'description_light intensity drizzle', 'description_light rain',
                                    'description_mist', 'description_moderate rain', 'description_overcast clouds',
                                    'description_scattered clouds', 'description_thunderstorm',
                                    'description_thunderstorm with rain', 'description_very heavy rain',
                                    'temp_mean_window_24', 'temp_std_window_24', 
                                    'feels_like_mean_window_24', 'feels_like_std_window_24']  


            # datetime_obj = datetime.datetime.fromisoformat(full_datetime)
            logger.info("Full date param: %s", full_datetime)
            matching_row = data[data['date'] == full_datetime]
            
            if matching_row.empty:
                  return jsonify({'error': 'No matching date found in the data'}), 404

            features = matching_row[feature_columns].values
            
            logger.info("features shape: %s", features.shape)

            features_reshaped = features.reshape((1, features.shape[0], features.shape[1]))
            
            # Make prediction
            prediction = model.predict(features_reshaped)
            predicted_active_power = prediction[0, 0]

            # Return result
            response = {
            'next_active_power': predicted_active_power
            }
            return jsonify(response)

      except Exception as e:
            logger.error("Error during prediction: %s", e)
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))