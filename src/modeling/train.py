import os
import math
import time
import joblib
import logging

import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam


import src.config.settings as s


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

save_path = 'models'
os.makedirs(save_path, exist_ok=True)

class EnergyModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.dates = self.X_train['date']
        self._X_train = self.X_train .drop(columns=['date'])

    def train_evaluate_arima(self):
        arima_predictions = []
        y_test_splits = []
        logger.info("Training ARIMA model")
        tscv = TimeSeriesSplit(n_splits=5)
        training_start_time = time.time()

        for idx, (train_index, test_index) in enumerate(tscv.split(self.X_train)):
            y_train_split, y_test_split = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

            model = ARIMA(y_train_split, order=(1, 1, 1))
            model_fit = model.fit()

            y_pred = model_fit.forecast(steps=len(y_test_split))
            arima_predictions.extend(y_pred)
            y_test_splits.extend(y_test_split)


            score_mae = mean_absolute_error(y_test_split, y_pred)
            score_rmse = math.sqrt(mean_squared_error(y_test_split, y_pred))

            logger.info(f"ARIMA Split {idx + 1}: MAE: {score_mae:.4f}, RMSE: {score_rmse:.4f}")


        training_time = time.time() - training_start_time
        joblib.dump(model_fit, os.path.join(save_path, 'arima_model.pkl'))

        return np.array(arima_predictions), training_time, np.array(y_test_splits), self.dates

    def train_evaluate_prophet(self):
        prophet_predictions = []
        logger.info("Training Prophet model")
        tscv = TimeSeriesSplit(n_splits=s.n_splits)
        self.X_train['ds'] = self.X_train['date']
        self.X_train['y'] = self.y_train
        self.X_train = self.X_train[['ds', 'y']]
        training_start_time = time.time()

        for idx, (train_index, test_index) in enumerate(tscv.split(self.X_train)):
            train_split = self.X_train.iloc[train_index]
            test_split = self.X_train.iloc[test_index]

            model = Prophet()
            model.fit(train_split)

            forecast = model.predict(test_split[['ds']])
            y_pred = forecast['yhat'].values
            prophet_predictions.extend(y_pred)

            score_mae = mean_absolute_error(test_split['y'], y_pred)
            score_rmse = math.sqrt(mean_squared_error(test_split['y'], y_pred))

            logger.info(f"Prophet Split {idx + 1}: MAE: {score_mae:.4f}, RMSE: {score_rmse:.4f}")

        training_time = time.time() - training_start_time
        joblib.dump(model, os.path.join(save_path, 'prophet_model.pkl'))

        return np.array(prophet_predictions), training_time


    def train_evaluate_lstm(self):
      lstm_predictions = []
      print("Training LSTM model")
      tscv = TimeSeriesSplit(n_splits=5)
      training_start_time = time.time()
      self.X_train = self.X_train.drop(columns=["date"])
      i = 1
      for idx, (train_index, test_index) in enumerate(tscv.split(self.X_train)):
          print("split: ",i)
          i+=1
          X_train_split, X_test_split = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
          y_train_split, y_test_split = self.y_train.iloc[train_index], self.y_train.iloc[test_index]

          X_train_split = X_train_split.values
          X_test_split = X_test_split.values

          timesteps = 1
          num_features = X_train_split.shape[1]
          X_train_reshaped = X_train_split.reshape((X_train_split.shape[0], timesteps, num_features))
          X_test_reshaped = X_test_split.reshape((X_test_split.shape[0], timesteps, num_features))
          y_train_split = y_train_split.to_numpy()

          model = Sequential()
          model.add(LSTM(100, input_shape=(timesteps, num_features)))
          model.add(Dense(1, activation='relu'))
          model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')


          with tf.device('/gpu:0'):
              model.fit(X_train_reshaped, y_train_split, epochs=100, batch_size=64, verbose=0
                        #, callbacks=[GPUUsageLogger()]
                        )

          y_pred = model.predict(X_test_reshaped).flatten()
          lstm_predictions.extend(y_pred)

          score_mae = mean_absolute_error(y_test_split, y_pred)
          score_rmse = math.sqrt(mean_squared_error(y_test_split, y_pred))

          print(f"LSTM Split {idx + 1}: MAE: {score_mae:.4f}, RMSE: {score_rmse:.4f}")

      training_time = time.time() - training_start_time
      model.save(os.path.join(save_path, 'lstm_model_2.h5'))

      return np.array(lstm_predictions), training_time
"""

dataset = EnergyDataset(data_path = "data/raw/data.csv")
dataset._perform_feature_engineering()
X_train, _, y_train, _ = dataset._train_test_split()

trainer = EnergyModelTrainer( X_train, y_train)

arima_predictions = trainer.train_evaluate_arima()
#prophet_predictions = trainer.train_evaluate_prophet()
lstm_predictions = trainer.train_evaluate_lstm()

logger.info("Training completed for all models")"""
