import logging

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam


from src.dataset import EnergyDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIMAPredictor:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def fit(self, y_train):
        self.model = ARIMA(y_train, order=self.order).fit()

    def predict(self, steps):
        logger.info(f"Predicting next {steps} steps using ARIMA")
        return self.model.forecast(steps=steps)

class ProphetPredictor:
    def __init__(self):
        self.model = Prophet()

    def fit(self, train_data):
        self.model.fit(train_data)

    def predict(self, future):
        logger.info(f"Predicting future using Prophet")
        forecast = self.model.predict(future)
        return forecast['yhat']

class LSTMPredictor:
    def __init__(self, timesteps=1, num_features=1, lstm_units=200, learning_rate=0.001):
        self.timesteps = timesteps
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(self.timesteps, self.num_features)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        X_train_reshaped = X_train.reshape((X_train.shape[0], self.timesteps, self.num_features))
        self.model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X_test):
        logger.info(f"Predicting using LSTM")
        X_test_reshaped = X_test.reshape((X_test.shape[0], self.timesteps, self.num_features))
        return self.model.predict(X_test_reshaped).flatten()

dataset = EnergyDataset(data_path = "data/raw/data.csv")
dataset._perform_feature_engineering()
X_train, X_test, y_train, y_test = dataset._train_test_split()


arima_predictor = ARIMAPredictor(order=(1, 1, 1))
arima_predictor.fit(y_train)
arima_future_predictions = arima_predictor.predict(steps=len(y_test))

prophet_predictor = ProphetPredictor()
prophet_predictor.fit()

X_test['ds'] = X_test['date']  

prophet_future_predictions = prophet_predictor.predict(X_test)

lstm_predictor = LSTMPredictor(timesteps=1, num_features=X_train.shape[1])
lstm_predictor.fit(X_train.values, y_train.values, epochs=10, batch_size=32)
lstm_future_predictions = lstm_predictor.predict(X_test.values)
