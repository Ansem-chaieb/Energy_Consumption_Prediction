
import pandas as pd

import holidays
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import timedelta

import src.config.constants as c


def seasonality_features(df, target, use_seasonality=True):

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    lags = [1, 8]
    for lag in lags:
        df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    return df

def datetime_features(data):

    data[c.DATE] = pd.to_datetime(data[c.DATE])
    

    data['year'] = data[c.DATE].dt.year
    data['month_of_year'] = data[c.DATE].dt.month
    data['week_of_year'] = data[c.DATE].dt.isocalendar().week
    data['day_of_year'] = data[c.DATE].dt.dayofyear
    data['day_of_week'] = data[c.DATE].dt.dayofweek
    data['hour_of_day'] = data[c.DATE].dt.hour
    data['is_weekend'] = (data[c.DATE].dt.dayofweek >= 5).astype(int)
    
    return data


def extract_holidays(df, country_code):
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be a DateTimeIndex.")

    start_date = df.index.min().date()
    end_date = df.index.max().date()

    country_holidays = holidays.CountryHoliday(country_code)
    holiday_dates = set()
    current_date = start_date
    while current_date <= end_date:
        if current_date in country_holidays:
            holiday_dates.add(current_date)
        current_date += timedelta(days=1)
    
    return holiday_dates

def holiday_features(df, country):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    holiday_dates = extract_holidays(df, country)
    df['is_holiday'] = df.index.to_series().dt.date.isin(holiday_dates).astype(int)
    df = df.reset_index()
    return df


def rolling_window_features(df, variables, windows, functions):
    for var in variables:
        for window in windows:
            for func in functions:
                col_name = f"{var}_{func}_window_{window}"
                if func == "mean":
                    df[col_name] = df[var].rolling(window=window).mean()
                elif func == "std":
                    df[col_name] = df[var].rolling(window=window).std()
    return df

def expanding_window_features(df, variables, functions):
    for var in variables:
        for func in functions:
            col_name = f"{var}_{func}_expanding"
            if func == "mean":
                df[col_name] = df[var].expanding().mean()
            elif func == "std":
                df[col_name] = df[var].expanding().std()

    return df

def window_features(df, variables):
          
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    windows = [24, 24 * 7, 24 * 365]  # Day, week, year
    functions = ["mean", "std"]
    df = rolling_window_features(df, variables, windows, functions)
    
    expanding_functions = ["mean", "std"]
    df = expanding_window_features(df, variables, expanding_functions)
    df = df.reset_index()
    return df

def preprocess_features(data, target, cat_features, date_feature=None):


    features = data.drop(columns=[target])
    target_col = data[target]
    date_feature_data = features[c.DATE]
    features = features.drop(columns=[c.DATE])


    cat_features_data = features[cat_features]
    cat_features_data = cat_features_data.fillna(cat_features_data.mode().iloc[0])
    

    encoder = OneHotEncoder(drop='first', )
    cat_encoded = encoder.fit_transform(cat_features_data).toarray()
    cat_feature_names = encoder.get_feature_names(cat_features)
    num_features_data = features.drop(columns=cat_features)
    num_features_data = num_features_data.fillna(num_features_data.mean())
    

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_features_data)
    num_feature_names = num_features_data.columns
    transformed_data = pd.DataFrame(
        pd.concat([pd.DataFrame(num_scaled, columns=num_feature_names), pd.DataFrame(cat_encoded, columns=cat_feature_names)], axis=1)
    )

    if date_feature_data is not None:
        transformed_data[c.DATE] = date_feature_data.values
    transformed_data[target] = target_col.values
    
    return transformed_data
