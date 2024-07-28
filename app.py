import os
import boto3
from io import StringIO

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
model_path = 'notebooks/models/lstm_model.h5'

def read_csv_from_s3(bucket_name, file_key):
      s3 = boto3.client('s3')
      response = s3.get_object(Bucket=bucket_name, Key=file_key)
      data = response['Body'].read().decode('utf-8')
      df = pd.read_csv(StringIO(data))
      return df

try:
    model = load_model(model_path)
    data = read_csv_from_s3(bucket_name, 'test_data.csv')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading model: %s", e)
    traceback.print_exc()

@app.route('/predict', methods=['GET'])
def predict():
      try:
            full_datetime = request.args.get('datetime')
            # Select features

            feature_columns = ['voltage', 'reactive_power', 'power_factor', 'temp', 'feels_like',
                              'temp_min', 'temp_max', 'pressure', 'humidity', 'speed', 'deg',
                              'active_power_lag_1', 'active_power_lag_8', 'year', 'month_of_year',
                              'week_of_year', 'day_of_year', 'day_of_week', 'hour_of_day',
                              'is_weekend', 'is_holiday', 'main_Clouds', 'main_Drizzle', 'main_Fog',
                              'main_Haze', 'main_Mist', 'main_Rain', 'main_Thunderstorm']  


            # datetime_obj = datetime.datetime.fromisoformat(full_datetime)
            logger.info("Full date param: %s", full_datetime)
            matching_row = data[data['datetime'] == full_datetime]
            
            if matching_row.empty:
                  return jsonify({'error': 'No matching date found in the data'}), 404

            features = matching_row[feature_columns].values
            
            logger.info("features shape: %s", features.shape)

            features_reshaped = features.reshape((1, features.shape[0], features.shape[1]))
            
            # Make prediction
            prediction = model.predict(features_reshaped)
            predicted_active_power = float(prediction[0, 0])

            # Return result
            response = {
            'active_power':  np.round(predicted_active_power* 10) / 10
            }
            return jsonify(response)

      except Exception as e:
            logger.error("Error during prediction: %s", e)
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))