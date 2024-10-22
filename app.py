import os
import boto3
from io import StringIO
from dotenv import load_dotenv

import numpy as np
import pandas as pd

import logging
import traceback
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

model_path = 'notebooks/models/lstm_model.h5'
bucket_name = 'my-energy-data-bucket'

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')

def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    return df


try:
    model = load_model(model_path, compile=False)
    data = read_csv_from_s3(bucket_name, 'test_data.csv')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading model: %s", e)
    traceback.print_exc()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        full_datetime = request_data.get('datetime')

        feature_columns = ['voltage', 'reactive_power', 'power_factor', 'temp', 'feels_like',
                           'temp_min', 'temp_max', 'pressure', 'humidity', 'speed', 'deg',
                           'active_power_lag_1', 'active_power_lag_8', 'year', 'month_of_year',
                           'week_of_year', 'day_of_year', 'day_of_week', 'hour_of_day',
                           'is_weekend', 'is_holiday', 'main_Clouds', 'main_Drizzle', 'main_Fog',
                           'main_Haze', 'main_Mist', 'main_Rain', 'main_Thunderstorm']

        logger.info("Full date param: %s", full_datetime)
        matching_row = data[data['datetime'] == full_datetime]
        
        if matching_row.empty:
            return jsonify({'error': 'No matching date found in the data'}), 404

        features = matching_row[feature_columns].values
        
        logger.info("Features shape: %s", features.shape)

        features_reshaped = features.reshape((1, features.shape[0], features.shape[1]))
        
        prediction = model.predict(features_reshaped)
        predicted_active_power = float(prediction[0, 0])

        response = {
            'active_power': np.round(predicted_active_power * 10) / 10
        }
        return jsonify(response)

    except Exception as e:
        logger.error("Error during prediction: %s", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
