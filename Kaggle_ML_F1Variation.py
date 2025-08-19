import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Load the results.csv file
f1_data = pd.read_csv("F1_variation/results.csv")
filtered_data = f1_data[f1_data['year'] >= 2010].copy()
print("Rows after filtering:", len(filtered_data))
print("")

# Clean: Replace \N with NaN
filtered_data.replace('\\N', pd.NA, inplace=True)

# Convert columns to numeric as needed
numeric_cols = ['grid', 'laps', 'milliseconds', 'fastestLap', 'rank'] # , 'positionOrder'
for col in numeric_cols:
    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

# Drop rows with NaN in key columns
filtered_data.dropna(subset=numeric_cols, inplace=True) #+ ['positionOrder']

# Show basic structure
print("F1 Dataset shape:", filtered_data.shape)
print(filtered_data.head())
print(filtered_data.columns)
print()

# Drop rows with missing values
filtered_data = filtered_data.dropna(axis=0)

# Simulate new features (you'll replace these with real ones later)
filtered_data['fastestLapTime'] = 90  # in seconds (placeholder)
filtered_data['pitStops'] = 2         # total pit stops (placeholder)
filtered_data['teammateGridDiff'] = 0 # difference from teammate's grid (placeholder)

# Combine numeric columns and simulated features
feature_cols = numeric_cols + ['fastestLapTime', 'pitStops', 'teammateGridDiff']

# Define target and features
y = filtered_data['points']
X = filtered_data[feature_cols]

# Show summary of input features
print(X.describe())
print(X.head())
print("")

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=1)

# Random Forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_valid)
rf_mae = mean_absolute_error(y_valid, rf_preds)

# Print predictions and actual values
print("Predictions:", rf_preds[:10])  # Show first 10 predictions
print("Actual:", y_valid[:10].values)          # Show first 10 actual values

# Evaluate model using Mean Absolute Error
print("Validation MAE:", rf_mae)

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_absolute_error')
mean_cv_mae = -cv_scores.mean()
print("Cross-validation Scores", cv_scores)
print("Mean Cross-validation Scores", mean_cv_mae, "\n")

# Simpler model for comparison
simple_model = DecisionTreeRegressor(max_depth=3, random_state=1)
simple_model.fit(X_train, y_train)
simple_preds = simple_model.predict(X_valid)
simple_mae = mean_absolute_error(y_valid, simple_preds)

# Print predictions and actual values
print("Predictions:", simple_preds[:10])  # Show first 10 predictions
print("Actual:", y_valid[:10].values)          # Show first 10 actual values

# Evaluate model using Mean Absolute Error
print("Validation MAE:", simple_mae, "\n")


# # Feature importance plot
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=True)
print("Feature Importance:",importance_df)

# Feature importance
plt.figure(figsize=(8, 5))
plt.barh(X.columns, rf_model.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()