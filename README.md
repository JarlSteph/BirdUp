# BirdUp – Scalable Bird Sighting Prediction System

## Final Project
**Course:** ID2223 / FID3020 HT25 – Scalable Machine Learning and Deep Learning  
**Project type:** Final Project  

**Authors:**
- Stefan Ivchenko
- Jarl Stephansson

---

## Project Overview

BirdUp is an end-to-end, scalable machine learning system for predicting **daily bird sighting probabilities** across Swedish administrative regions (*landskap*). The project focuses on two eagle species:

- **White-tailed eagle** (`whteag`)
- **Golden eagle** (`goleag`)

The system integrates dynamic data ingestion, feature engineering, neural network training, model versioning, daily inference, and a frontend for visualization.

---

## Dynamic Data Sources

### Weather Data
Historical and daily weather data is collected using the **geographical center of each Swedish region** to define latitude and longitude.

- Source: Meteo weather API
- Coverage: Daily data from **2011 to present**
- Data is updated continuously for inference

Weather features include:
- Wind speed
- Precipitation
- Temperature
- Categorical weather code

### Bird Observation Data
Bird observations are collected from **eBird**, a global citizen-science platform with continuously updated data.

- Species: white-tailed eagle and golden eagle
- Coverage: Historical data from **2011 to present**
- Observations are aggregated per region and day

Label construction:
- If sightings exist, we record the **number of birds observed**
- If no sightings exist, the day is treated as a **negative observation**

This allows the model to learn from both presence and absence of sightings.

---

## Prediction Problem

The goal is to predict whether eagles will be sighted in a given region on a given day.

For each combination of:
- region
- observation_date
- bird_type

the model outputs:
- a **probability** of a sighting
- a **binary prediction** based on a fixed threshold

Separate models are trained for each species to capture species-specific spatial and temporal patterns.

---

## Feature Engineering

All data sources are merged into a single daily feature table per region and bird type.

### Primary Keys
- `region` – Swedish administrative region (*landskap*)
- `observation_date` – calendar date (event time)
- `bird_type` – species identifier

### Weather Features
- `wind` – average daily wind speed (km/h)
- `rain` – total daily precipitation (mm)
- `weathercode` – categorical weather condition
- `temperature` – average daily air temperature (°C)

### Observation Features
- `observation_count` – number of birds observed
- `time_observations_started` – start time of observation effort

### Temporal Encoding
- **Year rebasing:**  
  The year is normalized such that 2011 = 0 and increases incrementally each year.
- **Month one-hot encoding:**  
  Twelve boolean features (`month_1` to `month_12`) indicate the observation month.

### Lag Features
To capture temporal dependencies, we include lagged features for the previous five days:

- `sighted_lag_1` to `sighted_lag_5` (binary indicator)
- `obs_count_lag_1` to `obs_count_lag_5` (number of birds observed)

---

## Feature Store and Model Registry (Hopsworks)

All feature-engineered data is stored in the **Hopsworks Feature Store**, enabling reproducible training and inference.

Trained models are stored and versioned in the **Hopsworks Model Registry**, allowing inference pipelines to reliably retrieve the correct model artifacts.

---

## Model Training

We train a feed-forward neural network (`BirdPercentModel`) using PyTorch. Separate models are trained for each species.

### Training Code
```python
goldag_model = BirdPercentModel(in_features=g_train_x.shape[1], hidden_layers=[32, 16, 1]).to(device=device)
goldag_model = train_model(
    g_train_x, g_train_y, g_val_x, g_val_y,
    goldag_model, num_epochs=6000, learning_rate=0.01, val=False
)

whteag_model = BirdPercentModel(in_features=w_train_x.shape[1], hidden_layers=[64, 32, 1]).to(device=device)
whteag_model = train_model(
    w_train_x, w_train_y, w_val_x, w_val_y,
    whteag_model, num_epochs=10000, learning_rate=0.01, val=False
)
```

After training, the model weights are uploaded to the **Hopsworks Model Registry** and versioned for later use in inference.

---

## Daily Inference Pipeline

Each day, the system performs the following steps:

1. Collect new weather data
2. Collect new bird observations
3. Generate feature-compatible rows and update lag features
4. Download trained models from the Hopsworks Model Registry
5. Load model weights into the PyTorch architectures
6. Perform inference per region and bird type
7. Upload prediction results to the Hopsworks Feature Store

The prediction output includes:
- `observation_date`
- `region`
- `bird_type`
- predicted probability
- binary prediction

---

## Frontend

A frontend application is included in the repository under the `Frontend/` directory.

The frontend visualizes:
- predicted sighting probabilities per region
- species-specific predictions
- the practical value of the ML pipeline outputs

---

## Repository Structure (High-Level)

- `Features/` – feature generation and data processing utilities
- `Models/` – neural network model definitions
- `inference_daily.py` – daily inference and prediction upload pipeline
- `Frontend/` – frontend application

---

## Summary

BirdUp demonstrates a complete scalable machine learning workflow:
- Dynamic real-world data sources
- Temporal feature engineering with lagged signals
- Neural network training and versioning
- Hopsworks-based MLOps
- Automated daily inference
- Frontend visualization of predictions

This project fulfills all requirements for the ID2223 / FID3020 final project.


to run it locally, do: 

***cd Frontend && npm run rev***
