# Energy Consumption Prediction Solution

This repository contains a solution for predicting household energy consumption based on historical data and weather information.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [API Usage](#api-usage)
5. [Streamlit Dashboard](#streamlit-dashboard)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)


## Project Overview

This project aims to predict the next value of the "active_power" column in a dataset of household energy consumption. The solution includes:

- Data preprocessing and feature engineering
- Model training and evaluation
- A Streamlit dashboard for data visualization and model insights
- Containerized model deployment
- A Flask API for serving predictions
- Cloud infrastructure design suggestion
- Considerations for production-scale operationalization

## Cloud Infrastructure Schema

Our proposed cloud infrastructure leverages AWS services to create a scalable, maintainable, and secure solution.

For a detailed diagram and explanation of the infrastructure, please refer to the [pdf](docs/Cloud_Infrastructure.md) file.

## Prerequisites

- Python 
- tensorflow
- streamlit 
- Docker
- flask

## Setup and Installation

1. Clone this repository:

`https://github.com/Ansem-chaieb/ve2max_assessment.git
cd energy-consumption-prediction`

2. Build and run the Docker container:

`docker-compose up --build`

3. Access the Flask API at `http://localhost:5000`
4. Access the Streamlit dashboard at `http://localhost:8501`

## API Usage

To get a prediction, send a POST request to the `/predict` endpoint:

`bash
curl -X POST -H "Content-Type: application/json" -d '{
"temperature": 25.5,
"humidity": 60,
"wind_speed": 5.2,
"previous_active_power": 1500
}' http://localhost:5000/predict`


## Streamlit Dashboard

The Streamlit dashboard provides interactive visualizations of the dataset and model performance. It includes:

Exploratory Data Analysis (EDA) charts
Feature importance plots
Model performance metrics
Real-time prediction interface

Access the dashboard at: http://localhost:8501

## Project Structure

energy-consumption-prediction/
├── api/
│   ├── app.py
│   ├── models.py
│   └── utils.py
├── dashboard/
│   └── streamlit_app.py
├── notebooks/
│   └── EDA.ipynb
├── tests/
│   └── test_api.py
├── docs/
│   ├── Cloud_Infrastructure.md
│   └── Operationalization.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

## Contributing Members

**Data Scientist: [Full Name](@slackHandle)**