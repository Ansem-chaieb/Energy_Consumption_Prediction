import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import streamlit as st
import pandas as pd

import boto3
from io import StringIO

from src.plots import create_active_power_histogram, create_box_plot, create_correlation_heatmap, create_count_plot, create_daily_avg_plot, create_energy_features_plot, create_feature_histograms, create_heatmap, create_weather_feature_plots

st.set_page_config(layout="wide")


@st.cache_data
def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    return df

def load_data():
    data = pd.read_csv("data/raw/data.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

bucket_name = 'my-energy-data-bucket'
file_key = 'data.csv'

data = read_csv_from_s3(bucket_name, file_key)


def calculate_summary_stats(data):
    num_observations = len(data)
    data['date_'] = pd.to_datetime(data['date'])
    num_days = (data['date_'].max() - data['date_'].min()).days + 1
    highest_active_power = data['active_power'].max()
    lowest_active_power = data['active_power'].min()
    return num_observations,  num_days, highest_active_power, lowest_active_power


def load_css():
    st.markdown("""
    <style>
    .stApp {
        display: flex;
        justify-content: flex-end;
        padding: 1rem;
    }
    .square {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 350px;
        height: 150px;
        background-color: #FFFFFF;
        border: 2px solid #FFF;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    .square-content {
        font-size: 70px;
    }
    </style>
    """, unsafe_allow_html=True)


def create_summary_squares(num_observations, num_days, highest_active_power, lowest_active_power):
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown(f"""
        <div class="square">
            <div class="square-content">
                <div class="right-content"><p style="font-size: 15px;">Observations:</div>
                <div class="left-content">{num_observations}</div>
            </div>
        </div>
        <div class="square">
            <div class="square-content">
                <div class="right-content"><p style="font-size: 15px;">Highest Active Power:</p></div>
                <div class="left-content">{highest_active_power:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="square">
            <div class="square-content">
                <div class="right-content"><p style="font-size: 15px;">Number of Days:</div>
                <div class="left-content">{num_days}</div>
            </div>
        </div>
        <div class="square">
            <div class="square-content">
                <div class="right-content"><p style="font-size: 15px;">Lowest Active Power:</p></div>
                <div class="left-content">{lowest_active_power:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3: 
        fig_hist = create_active_power_histogram(data)
        st.plotly_chart(fig_hist, use_container_width=True)

data_dict = {
    "Group Name": ["Timestamp", "Energy Metrics", "Energy Metrics", "Energy Metrics", "Energy Metrics", "Energy Metrics", "Energy Metrics", 
                   "Weather Conditions", "Weather Conditions", "Weather Conditions", "Weather Conditions", "Weather Conditions", "Weather Conditions", 
                   "Weather Conditions", "Weather Conditions", "Weather Conditions", "Weather Conditions",
                   "Forecasted Weather", "Forecasted Weather"],
    "Column Name": ["date", "active_power", "current", "voltage", "reactive_power", "apparent_power", "power_factor",
                    "main", "description", "temp", "feels_like", "temp_min", "temp_max", "pressure", "humidity", "speed", "deg",
                    "temp_t+1", "feels_like_t+1"],
    "Description": [
        "Date and time of the recorded data point",
        "Active power consumed by the residence",
        "Electrical current flowing through the system",
        "Electrical potential difference in the system",
        "Reactive power in the system",
        "Total power in the system, combining both active and reactive power",
        "Ratio of active power to apparent power",
        "Main categorical weather condition",
        "Detailed categorical weather condition",
        "Ambient temperature at the time of recording",
        "Perceived temperature considering wind and humidity",
        "Minimum temperature recorded during the interval",
        "Maximum temperature recorded during the interval",
        "Atmospheric pressure",
        "Relative humidity",
        "Wind speed",
        "Wind direction in degrees",
        "Forecasted temperature for the next day",
        "Forecasted 'feels like' temperature for the next day"
    ]
}
df = pd.DataFrame(data_dict)

# Define colors for each group
group_colors = {
    "Timestamp": "background-color: #FFDDC1",  # light orange
    "Energy Metrics": "background-color: #C1FFC1",  # light green
    "Weather Conditions": "background-color: #C1D4FF",  # light blue
    "Forecasted Weather": "background-color: #F8C1FF"  # light pink
}
def highlight_groups(row):
    return [group_colors.get(row["Group Name"], "")] * len(row)

def main():
    st.title("Household Energy Consumption Dashboard")
    section = st.radio('Select Section', ['Data description', 'EDA', 'Model results'], horizontal=True)
    if section == 'Data description':
        styled_df = df.style.apply(highlight_groups, axis=1)
        st.dataframe(styled_df.set_table_styles(
            [{'selector': 'thead th', 'props': [('background-color', '#f4f4f4')]}]
        ))
        st.markdown("The data is available [here](https://data.mendeley.com/datasets/tvhygj8rgg/1).")

    elif section == 'EDA':
        num_observations,  num_days, highest_active_power, lowest_active_power = calculate_summary_stats(data)
        load_css()
        create_summary_squares(num_observations, num_days, highest_active_power, lowest_active_power)
        fig_daily_avg = create_daily_avg_plot(data)
        st.plotly_chart(fig_daily_avg, use_container_width=True)

        st.subheader("Energy Consumption Features:")
        with st.expander("Energy Consumption Features"):
            col1, col2 = st.columns([3, 2])
            with col1:
                fig_line = create_energy_features_plot(data)
                st.plotly_chart(fig_line, use_container_width=True)
            with col2:
                features = ['active_power', 'current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor']
                fig_heatmap = create_correlation_heatmap(data, features)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            features = ['current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor']
            fig_histograms = create_feature_histograms(data, features)
            cols = st.columns(len(features))
            for col, feature in zip(cols, features):
                with col:
                    st.plotly_chart(fig_histograms[feature], use_container_width=True)
        
        st.subheader("Weather Condition Features:")
        with st.expander("Weather Condition Features"):
            for category_feature in ['main', 'description']:
                col1, col2 = st.columns([1, 1])
                with col1:
                    bar_chart = create_count_plot(data, category_feature)
                    st.pyplot(bar_chart)
                with col2:
                    box_plot = create_box_plot(data, category_feature)
                    st.pyplot(box_plot)
            
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Temperature and Weather Metrics:")
            with st.expander("Temperature and Weather Metrics"):
                features1 = ['active_power', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity']
                corr_matrix1 = data[features1].corr()
                heatmap1 = create_heatmap(corr_matrix1, 'Correlation Matrix of Features 1')
                st.plotly_chart(heatmap1, use_container_width=True)
        
        with col2:
            st.subheader("Speed Features:")
            with st.expander("Speed Features"):
                features2 = ['active_power', 'speed', 'deg']
                corr_matrix2 = data[features2].corr()
                heatmap2 = create_heatmap(corr_matrix2, 'Correlation Matrix of Features 2')
                st.plotly_chart(heatmap2, use_container_width=True)
        
        features = ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'speed', 'deg']
        figures, summary_df = create_weather_feature_plots(data, features)
        
        col1, col2 = st.columns([1,1])
        for i, feature in enumerate(features):
            with col1 if i < len(features)//2 else col2:
                st.image(figures[feature], use_column_width=True)
                st.write(f"### {feature.replace('_', ' ').title()}")
                st.dataframe(pd.DataFrame(summary_df.loc[[feature]]).T, use_container_width=True)
    else:
              pass

if __name__ == "__main__":
    main()