import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO


color_scheme = px.colors.qualitative.Dark2

def create_daily_avg_plot(data):
    daily_data = data['active_power'].resample('D').mean().reset_index()
    fig = px.line(daily_data, x='date', y='active_power',
                  title='Daily Average Active Power over Time: 2022-11-05 - 2024-01-05',
                  labels={'active_power': 'Active Power (kW)', 'date': 'Date'},
                  color_discrete_sequence=[color_scheme[0]])
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def create_active_power_histogram(data):
    fig_hist = px.histogram(data, x='active_power',
                            title='Distribution of Active Power',
                            labels={'active_power': 'Active Power (kW)'},
                            color_discrete_sequence=[color_scheme[1]])
    fig_hist.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    return fig_hist


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


def create_count_plot(data, category_feature):
    plt.figure(figsize=(12, 6))

    sns.countplot(x=category_feature, data=data, palette='viridis')
    plt.title(f'Count of Observations by {category_feature}')
    plt.xlabel(category_feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


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


def create_box_plot(data, category_feature):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=category_feature, y='active_power', data=data, showfliers=False)
    plt.title(f'Effect of {category_feature} on Active Power')
    plt.xlabel(category_feature)
    plt.ylabel('Active Power (kW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


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