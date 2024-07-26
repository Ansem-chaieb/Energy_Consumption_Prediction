import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set page configuration
st.set_page_config(layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("data/raw/data.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

data = load_data()

color_scheme =px.colors.qualitative.Dark2

# Calculate summary statistics
def calculate_summary_stats(data):
    num_observations = len(data)
    date_range_start = data.index.min().strftime('%Y-%m-%d')
    date_range_end = data.index.max().strftime('%Y-%m-%d')
    num_days = (data.index.max() - data.index.min()).days + 1
    highest_active_power = data['active_power'].max()
    lowest_active_power = data['active_power'].min()
    return num_observations, date_range_start, date_range_end, num_days, highest_active_power, lowest_active_power

# Custom CSS
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

# Create summary squares
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

# Create daily average active power plot
# Create daily average active power plot
def create_daily_avg_plot(data):
    daily_data = data['active_power'].resample('D').mean().reset_index()
    fig = px.line(daily_data, x='date', y='active_power',
                  title='Daily Average Active Power over Time: 2022-11-05 - 2024-01-05',
                  labels={'active_power': 'Active Power (kW)', 'date': 'Date'},
                  color_discrete_sequence=[color_scheme[0]])
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    return fig

# Create active power distribution histogram
def create_active_power_histogram(data):
    fig_hist = px.histogram(data, x='active_power',
                            title='Distribution of Active Power',
                            labels={'active_power': 'Active Power (kW)'},
                            color_discrete_sequence=[color_scheme[1]])
    fig_hist.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    return fig_hist

# Create energy consumption features plot
def create_energy_features_plot(data):
    daily_data = data[['active_power', 'current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor']].resample('D').mean()
    fig_line = go.Figure()
    features = ['active_power', 'current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor']
    for i, feature in enumerate(features):
        fig_line.add_trace(go.Scatter(x=daily_data.index, y=daily_data[feature], mode='lines', name=feature, line=dict(color=color_scheme[i % len(color_scheme)])))
    fig_line.update_layout(
        title='Daily Averages of Features over Time',
        xaxis_title='Date',
        yaxis_title='Value',
        height=500,
        width=800,
        legend_title='Features'
    )
    return fig_line

# Create correlation heatmap
def create_correlation_heatmap(data, features):
    corr_matrix = data[features].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    fig_heatmap.update_layout(
        title='Correlation Matrix of Features',
        xaxis_title='Features',
        yaxis_title='Features',
        height=500,
        width=400
    )
    return fig_heatmap

# Create feature histograms
def create_feature_histograms(data, features):
    fig_histograms = {}
    for i, feature in enumerate(features):
        fig_histograms[feature] = go.Figure(data=[go.Histogram(x=data[feature], marker_color=color_scheme[i % len(color_scheme)])])
        fig_histograms[feature].update_layout(
            title=f'{feature.replace("_", " ").title()} Distribution',
            xaxis_title=feature.replace("_", " ").title(),
            yaxis_title='Count',
            height=300,
            width=300
        )
    return fig_histograms



# Create count plot
def create_count_plot(data, category_feature):
    plt.figure(figsize=(12, 6))

    sns.countplot(x=category_feature, data=data, palette='viridis')
    plt.title(f'Count of Observations by {category_feature}')
    plt.xlabel(category_feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

# Create box plot
def create_box_plot(data, category_feature):
    plt.figure(figsize=(12, 6))
    sns.set_palette(sns.color_palette(color_scheme))
    sns.boxplot(x=category_feature, y='active_power', data=data, showfliers=False)
    plt.title(f'Effect of {category_feature} on Active Power')
    plt.xlabel(category_feature)
    plt.ylabel('Active Power (kW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

# Create box plot
def create_box_plot(data, category_feature):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=category_feature, y='active_power', data=data, showfliers=False)
    plt.title(f'Effect of {category_feature} on Active Power')
    plt.xlabel(category_feature)
    plt.ylabel('Active Power (kW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

# Create heatmap
def create_heatmap(corr_matrix, title):
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Features',
        yaxis_title='Features',
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=-45)
    )
    return fig

# Create line plots for weather features
def create_weather_feature_plots(data, features):
    figures = {}
    summary_stats = {}
    for i, feature in enumerate(features):
        plt.figure(figsize=(10, 6))
        daily_data = data[feature].resample('D').mean()
        color_scheme = ["#1b9e77","#d95f02" ,"#7570b3", "#e7298a","#66a61e" , "#e6ab02"]  # Define colors as hex codes

        plt.plot(daily_data.index, daily_data, label=feature, color=color_scheme[i % len(color_scheme)])        
        plt.title(f'Daily Average {feature.replace("_", " ").title()} over Time')
        plt.xlabel('Date')
        plt.ylabel(feature.replace("_", " ").title())
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        figures[feature] = buf
        plt.close()
        
        col_data = data[feature]
        summary_stats[feature] = {
            'Mean': col_data.mean(),
            'Median': col_data.median(),
            'Std Dev': col_data.std(),
            'Variance': col_data.var()
        }
    summary_df = pd.DataFrame(summary_stats).T
    return figures, summary_df


# Main function
def main():
    st.title("Household Energy Consumption Dashboard")
    
    # Load data and calculate summary statistics
    
    num_observations, date_range_start, date_range_end, num_days, highest_active_power, lowest_active_power = calculate_summary_stats(data)
    
    # Load custom CSS
    load_css()
    
    # Create summary squares
    create_summary_squares(num_observations, num_days, highest_active_power, lowest_active_power)
    
    # Create and display daily average active power plot
    fig_daily_avg = create_daily_avg_plot(data)
    st.plotly_chart(fig_daily_avg, use_container_width=True)

        
    
    # Energy Consumption Features
    st.subheader("Energy Consumption Features:")
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_line = create_energy_features_plot(data)
        st.plotly_chart(fig_line, use_container_width=True)
    with col2:
        features = ['active_power', 'current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor']
        fig_heatmap = create_correlation_heatmap(data, features)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Feature histograms
    features = ['current', 'voltage', 'reactive_power', 'apparent_power', 'power_factor']
    fig_histograms = create_feature_histograms(data, features)
    cols = st.columns(len(features))
    for col, feature in zip(cols, features):
        with col:
            st.plotly_chart(fig_histograms[feature], use_container_width=True)
    
    # Weather Condition Features
    st.subheader("Weather Condition Features:")
    for category_feature in ['main', 'description']:
        col1, col2 = st.columns([1, 1])
        with col1:
            bar_chart = create_count_plot(data, category_feature)
            st.pyplot(bar_chart)
        with col2:
            box_plot = create_box_plot(data, category_feature)
            st.pyplot(box_plot)
    
    # Temperature and Weather Metrics
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Temperature and Weather Metrics:")
        features1 = ['active_power', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity']
        corr_matrix1 = data[features1].corr()
        heatmap1 = create_heatmap(corr_matrix1, 'Correlation Matrix of Features 1')
        st.plotly_chart(heatmap1, use_container_width=True)
    
    with col2:
        st.subheader("Forecasted Temperature Features:")
        features2 = ['active_power', 'speed', 'deg']
        corr_matrix2 = data[features2].corr()
        heatmap2 = create_heatmap(corr_matrix2, 'Correlation Matrix of Features 2')
        st.plotly_chart(heatmap2, use_container_width=True)
    
    # Weather feature plots
    features = ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'speed', 'deg']
    figures, summary_df = create_weather_feature_plots(data, features)
    
    col1, col2 = st.columns([1,1])
    for i, feature in enumerate(features):
        with col1 if i < len(features)//2 else col2:
            st.image(figures[feature], use_column_width=True)
            st.write(f"### {feature.replace('_', ' ').title()}")
            st.dataframe(pd.DataFrame(summary_df.loc[[feature]]).T, use_container_width=True)

if __name__ == "__main__":
    main()