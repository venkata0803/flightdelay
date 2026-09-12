# Flight Delay Prediction Using Machine Learning

A machine learning project for predicting flight delays using historical flight and weather data. The project uses a two-stage approach: first predicting whether a flight will be delayed, and then estimating the duration of the delay in minutes.

## Project Overview

Flight delays can cause inconvenience to passengers and financial losses for airlines. This project aims to build a predictive system that can:

1. Predict whether a flight will be delayed by 15 minutes or more.
2. Predict the expected duration of the arrival delay in minutes.

The project combines historical flight data with weather conditions from **15 major U.S. airports** for the years **2016 and 2017**. :contentReference[oaicite:1]{index=1}

## Key Features

- Flight delay classification
- Flight delay duration prediction
- Integration of flight and weather data
- Data preprocessing and feature selection
- Handling class imbalance using SMOTE
- Multiple classification models
- Multiple regression models
- Two-stage classification and regression pipeline
- Performance evaluation using multiple metrics
- Analysis of model performance across different delay intervals

## Dataset

The project uses two main datasets:

### Flight Data

The flight data contains records for 2016 and 2017 and includes information such as:

- Flight date
- Departure and arrival times
- Scheduled departure and arrival times
- Departure delay
- Arrival delay
- Origin airport
- Destination airport
- Delay indicators

The main target variables are:

- `ArrDel15` – Indicates whether the arrival delay is 15 minutes or more.
- `ArrDelayMinutes` – Represents the arrival delay duration in minutes. :contentReference[oaicite:2]{index=2}

### Weather Data

Weather information is filtered to 2016 and 2017 and includes:

- Wind speed
- Wind direction
- Weather code
- Precipitation
- Visibility
- Pressure
- Cloud cover
- Dew point
- Wind gust
- Temperature
- Wind chill
- Humidity
- Date and time
- Airport

The selected airports are:

```text
ATL, CLT, DEN, DFW, EWR,
IAH, JFK, LAS, LAX, MCO,
MIA, ORD, PHX, SEA, SFO
```

:contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

## Data Preprocessing

The flight and weather datasets are filtered and merged to create a unified dataset.

### Processing Steps

1. Filter flight and weather data for the selected airports.
2. Keep data from 2016 and 2017.
3. Select relevant flight and weather features.
4. Convert date-related information into numerical form.
5. Match weather information with flight information using:
   - Airport
   - Date
   - Scheduled departure time
6. Create the final dataset for machine learning. :contentReference[oaicite:5]{index=5}

## Features Used

The following features are used for both classification and regression:

```text
CRSArrTime
WindSpeedKmph
WindDirDegree
WeatherCode
precipMM
Visibilty
Pressure
Cloudcover
DewPointF
WindGustKmph
tempF
WindChillF
Humidity
Year
Quarter
Month
DayofMonth
CRSDepTime
DepDelayMinutes
```

:contentReference[oaicite:6]{index=6}

## Classification

The classification stage predicts whether a flight will be delayed by 15 minutes or more.

### Target

```text
ArrDel15
```

where:

```text
0 → On Time
1 → Delayed
```

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance between delayed and on-time flights. :contentReference[oaicite:7]{index=7}

### Classification Models

The following models are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees Classifier
- XGBoost

### Classification Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.85 |
| Decision Tree | 0.89 |
| Random Forest | 0.90 |
| Extra Trees | 0.90 |
| XGBoost | 0.92 |

XGBoost achieved the highest standalone classification accuracy of **0.92** among the models evaluated. :contentReference[oaicite:8]{index=8}

## Classification Metrics

The project evaluates classification performance using:

- Accuracy
- Precision
- Recall
- F1-Score
- Support

These metrics provide a more detailed evaluation, particularly because the dataset contains class imbalance. :contentReference[oaicite:9]{index=9}

## Regression

The regression stage predicts the duration of the flight arrival delay.

### Target

```text
ArrDelayMinutes
```

The same set of flight and weather features used for classification is used for regression. :contentReference[oaicite:10]{index=10}

### Regression Models

The following models are evaluated:

- Linear Regression
- Extra Trees Regressor
- Random Forest Regressor
- XGBoost Regressor

### Regression Results

| Model | R² Score | MSE | MAE | RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.92 | 133.22 | 5.55 | 11.54 |
| Extra Trees | 0.93 | 128.63 | 5.83 | 11.34 |
| Random Forest | 0.93 | 128.02 | 5.77 | 11.31 |
| XGBoost | 0.87 | 217.15 | 5.93 | 14.74 |

Random Forest Regressor achieved the lowest MSE and RMSE among the evaluated regression models, while Random Forest and Extra Trees both achieved an R² score of **0.93**. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

## Two-Stage Machine Learning Pipeline

The main approach combines classification and regression into a two-stage pipeline.

```text
                 Flight + Weather Data
                          ↓
                 Data Preprocessing
                          ↓
                    Classification
                          ↓
                 ┌────────┴────────┐
                 │                 │
              On Time           Delayed
                 │                 │
                 ↓                 ↓
                End           Regression
                                   ↓
                         Delay Duration Prediction
                                   ↓
                              Final Output
```

### Stage 1 - Classification

The classifier predicts:

```text
ArrDel15
```

- `0` → Flight is not delayed
- `1` → Flight is delayed

### Stage 2 - Regression

If the classification result is `1`, the regression model predicts:

```text
ArrDelayMinutes
```

This approach allows the system to first determine whether a delay is likely and then estimate its duration. :contentReference[oaicite:13]{index=13}

## Pipeline Model Combinations

The project evaluates combinations of:

```text
5 Classifiers × 4 Regressors = 20 Pipelines
```

The report identifies the best-performing pipeline as:

```text
XGBoost Classifier → Random Forest Regressor
```

with reported performance of:

```text
Classification Accuracy = 95%
Regression R² Score = 0.95
```

This combination was identified as the most reliable pipeline in the project. :contentReference[oaicite:14]{index=14}

## Regression Analysis

The regression performance is also analyzed across different delay intervals.

| Delay Interval | Frequency | R² Score | RMSE |
|---|---:|---:|---:|
| 15–100 minutes | 323,567 | 0.645 | 13.11 |
| 100–200 minutes | 48,955 | 0.707 | 14.62 |
| 200–500 minutes | 14,232 | 0.934 | 16.88 |
| 500–1000 minutes | 1,129 | 0.996 | 8.55 |
| 1000–2000 minutes | 173 | 0.994 | 12.75 |

The analysis shows stronger model performance for longer delay intervals, while shorter delays have more variation in prediction. :contentReference[oaicite:15]{index=15}

## Evaluation Metrics

### Classification

**Accuracy**

Measures the proportion of correctly classified instances.

**Precision**

Measures how many predicted positive instances are actually positive.

**Recall**

Measures how many actual positive instances are correctly identified.

**F1-Score**

Balances precision and recall and is useful when dealing with imbalanced datasets. :contentReference[oaicite:16]{index=16}

### Regression

**R² Score**

Measures how well the input variables explain the variance in the target variable. Higher values indicate a better fit.

**MSE**

Measures the average squared difference between actual and predicted values.

**MAE**

Measures the average absolute difference between actual and predicted values.

**RMSE**

Measures the standard deviation of prediction errors and gives greater weight to larger errors.

For MSE, MAE, and RMSE, lower values indicate better performance. :contentReference[oaicite:17]{index=17}

## Actual vs Predicted Analysis

The project also compares the distribution of actual and predicted arrival delays.

The analysis shows that:

- Most delays occur within the 0–100 minute range.
- The model closely follows the actual delay distribution.
- There are some deviations in the 100–200 minute range.
- Predictions align more closely with actual values for longer delays.
- The model performs particularly well for moderate to extreme delays. :contentReference[oaicite:18]{index=18}

## Project Workflow

```text
1. Load Flight Data
        ↓
2. Load Weather Data
        ↓
3. Filter Airports and Years
        ↓
4. Clean and Preprocess Data
        ↓
5. Merge Flight + Weather Data
        ↓
6. Feature Selection
        ↓
7. Train-Test Split
        ↓
8. Apply SMOTE
        ↓
9. Train Classification Models
        ↓
10. Select Classification Model
        ↓
11. Train Regression Models
        ↓
12. Select Regression Model
        ↓
13. Build Two-Stage Pipeline
        ↓
14. Evaluate Predictions
```

## Tools and Libraries

The project was developed using Python-based machine learning tools and environments, including:

- Python
- Google Colab
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- SMOTE

The project uses these tools for data processing, visualization, model training, evaluation, and pipeline construction. :contentReference[oaicite:19]{index=19}

## Project Structure

A typical repository structure can be organized as:

```text
Flight-Delay-Prediction/
│
├── data/
│   ├── flight_data/
│   └── weather_data/
│
├── notebooks/
│   └── flight_delay_prediction.ipynb
│
├── README.md
└── requirements.txt
```

> The exact folder and file structure may vary depending on the implementation.

## Results

The project demonstrates that combining flight information with weather conditions can be used to predict both the occurrence and duration of flight delays.

### Best Reported Pipeline

```text
XGBoost Classifier
        ↓
Random Forest Regressor
```

Reported performance:

```text
Classification Accuracy: 95%
Regression R² Score: 0.95
```

The project also demonstrates the importance of handling class imbalance using SMOTE and evaluating models using multiple performance metrics. :contentReference[oaicite:20]{index=20}

## Conclusion

This project develops a two-stage machine learning system for flight delay prediction.

The classification stage predicts whether a flight will experience a delay, while the regression stage estimates the duration of that delay. By combining historical flight information with weather conditions, the system provides a more comprehensive approach to flight delay prediction.

The results demonstrate strong predictive performance and potential applications in:

- Flight delay risk assessment
- Airline scheduling
- Resource allocation
- Operational planning
- Passenger information systems

Future improvements could include integrating real-time air traffic information, airport congestion data, and airline-specific delay patterns. :contentReference[oaicite:21]{index=21}

## License

This project was developed for educational and academic purposes.
