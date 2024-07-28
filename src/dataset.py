import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import pandas as pd
import logging


import src.config.constants as c
import src.config.settings as s
from src.features import (
    seasonality_features,
    datetime_features,
    holiday_features,
    window_features,
    preprocess_features
)

logger = logging.getLogger(__name__)

class EnergyDataset:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self._data = self._data = pd.read_csv(self.data_path)
        used_features = ['date', 'active_power', 'voltage', 'reactive_power',
       'power_factor', 'temp',
       'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'speed',
       'deg','main', 'description',
       ]

        self._data = self._data[used_features]

    def __getitem__(self, idx: int) -> pd.Series:
        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self._data.iloc[idx]

    def __len__(self):
        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return len(self._data)

    def _perform_feature_engineering(self):
        self._data = seasonality_features(self._data, 'active_power')
        self._data = datetime_features(self._data)
        self._data = holiday_features(self._data, country='MX')

        self._data = window_features(self._data, variables=['temp','feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'speed','deg'])
        self._data = preprocess_features(self._data, "active_power", cat_features=['main', 'description'])
        column_names = self._data.columns.tolist()
        print(column_names)
        self._data = self._data[['voltage', 'reactive_power','power_factor', 'active_power',
                               'temp', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity',
                               'speed', 'deg', 'active_power_lag_1', 'active_power_lag_8', 'year',
                               'month_of_year', 'week_of_year', 'day_of_year', 'day_of_week', 'hour_of_day',
                               'is_weekend', 'is_holiday', 'main_Clouds', 'main_Drizzle', 'main_Fog',
                               'main_Haze', 'main_Mist', 'main_Rain', 'main_Thunderstorm','date',
                               #'description_clear sky', 'description_drizzle', 'description_few clouds',
                               #'description_fog', 'description_haze', 'description_heavy intensity rain',
                               #'description_light intensity drizzle', 'description_light rain',
                               #'description_mist', 'description_moderate rain', 'description_overcast clouds',
                               #'description_scattered clouds', 'description_thunderstorm',
                               #'description_thunderstorm with rain', 'description_very heavy rain', 'date',
                               #'active_power', 'day',  'temp_mean_window_24', 'temp_std_window_24',
                               #'feels_like_mean_window_24', 'feels_like_std_window_24'
                               ]]

    def _train_test_split(self, test_size: float = 0.2) :
        if self._data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self._data['date'] = pd.to_datetime(self._data['date'])
        self._data = self._data.sort_values(by='date')
        self._data['day'] = self._data['date'].dt.date
        unique_days = self._data['day'].unique()

        train_days = int(len(unique_days) * (1 - test_size))
        train_day_set = unique_days[:train_days]
        test_day_set = unique_days[train_days:]

        train_data = self._data[self._data['day'].isin(train_day_set)]
        test_data = self._data[self._data['day'].isin(test_day_set)]



        X_train = train_data.drop(columns=['active_power', "day"])
        y_train = train_data['active_power']
        X_test = test_data.drop(columns=['active_power', "day"])
        y_test = test_data['active_power']

        return X_train, X_test, y_train, y_test
    
    
"""dataset = EnergyDataset(data_path = "./data/raw/data.csv")
dataset._perform_feature_engineering()
#X_train, X_test, y_train, y_test = dataset._train_test_split()"""
