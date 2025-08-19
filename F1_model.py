import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Load lap telemetry data
f1_data = pd.read_csv("F1_variation/data/f1_lap_telemetry_2022_2025.csv")

# Convert numeric columns
numeric_cols = [
    'grid', 'finish_position', 'lap_number', 'lap_time_sec',
    'sector1_time', 'sector2_time', 'sector3_time',
    'speed_i1', 'speed_i2', 'speed_fl'
]
for col in numeric_cols:
    if col in f1_data.columns:
        f1_data[col] = pd.to_numeric(f1_data[col], errors='coerce')

# Remove missing key values
f1_data.dropna(subset=['grid', 'finish_position', 'lap_time_sec'], inplace=True)

# Average pit stop lap time
f1_data['pit_lap_time'] = f1_data.apply(
    lambda row: row['lap_time_sec'] if row['is_pit_lap'] == 1 else pd.NA, axis=1
)

# Group to race-level
race_level_data = (
    f1_data
    .groupby(['year', 'race', 'driver'])
    .agg({
        'grid': 'first',
        'finish_position': 'first',
        'lap_number': 'max',
        'lap_time_sec': ['mean', 'min'],
        'sector1_time': ['mean', 'std'],
        'sector2_time': ['mean', 'std'],
        'sector3_time': ['mean', 'std'],
        'speed_i1': 'mean',
        'speed_i2': 'mean',
        'speed_fl': 'mean',
        'points': 'first',
        'is_pit_lap': 'sum',
        'pit_lap_time': 'mean'
    })
    .reset_index()
)

# Flatten multi-level column names
race_level_data.columns = ['_'.join(col).strip('_') for col in race_level_data.columns.values]

# Fastest lap flag
race_min_lap_times = race_level_data.groupby(['year', 'race'])['lap_time_sec_min'].transform('min')
race_level_data['fastest_lap_flag'] = (race_level_data['lap_time_sec_min'] == race_min_lap_times).astype(int)

# ------------------------
# Features & target
# ------------------------
y = race_level_data['points_first']
feature_cols = [
    c for c in race_level_data.columns 
    if c not in ['year', 'race', 'driver', 'points_first', 'finish_position_first']
]
X = race_level_data[feature_cols]

# ------------------------
# Preprocessing
# ------------------------
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ------------------------
# Train-test split
# ------------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=1)

# ------------------------
# Decision Tree
# ------------------------
dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(random_state=1))
])
dt_pipeline.fit(X_train, y_train)
dt_model = dt_pipeline.named_steps['model']
dt_preds = np.round(dt_pipeline.predict(X_valid)).astype(int)
dt_mae = mean_absolute_error(y_valid, dt_preds)
print("Decision Tree Model Prediction vs Actual Values & Validation MAE & CV Scores")
print("Predictions:", dt_preds[:10])
print("Actual:", y_valid[:10].values)
print("MAE:", dt_mae)
print("Mean CV MAE:", -cross_val_score(dt_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error').mean())
print()

# ------------------------
# Random Forest
# ------------------------
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=1))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = np.round(rf_pipeline.predict(X_valid)).astype(int)
rf_mae = mean_absolute_error(y_valid, rf_preds)
print("Random Forest Model Prediction vs Actual Values & Validation MAE & CV Scores")
print("Predictions:", rf_preds[:10])
print("Actual:", y_valid[:10].values)
print("MAE:", rf_mae)
print("Mean CV MAE:", -cross_val_score(rf_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error').mean())
print()

# ------------------------
# XGBoost
# ------------------------
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(enable_categorical=False, random_state=42))
])
xgb_pipeline.fit(X_train, y_train)
xgb_preds = np.round(xgb_pipeline.predict(X_valid)).astype(int)
xgb_mae = mean_absolute_error(y_valid, xgb_preds)
print("XGBoost Model Prediction vs Actual Values & Validation MAE & CV Scores")
print("Predictions:", xgb_preds[:10])
print("Actual:", y_valid[:10].values)
print("MAE:", xgb_mae)
print("Mean CV MAE:", -cross_val_score(xgb_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error').mean())
print()

# ------------------------
# Feature Importances
# ------------------------

# ------------------------
# Decision Tree
# ------------------------
print("Feature Importance of the Decision Tree Model")
feature_importances = dt_model.feature_importances_
if len(feature_importances) != len(X.columns):
    # Use the processed feature names after encoding/scaling
    feature_names = [f'feature_{i}' for i in range(len(feature_importances))]
else:
    feature_names = X.columns

# Now create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
})
importance_df = importance_df.sort_values(by='importance', ascending=True)
print("Feature Importance:",importance_df)

plt.figure(figsize=(8, 5))
plt.barh(feature_cols, dt_model.feature_importances_)
plt.title("Decision Tree Model Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show(block=True)
print("")

# ------------------------
# Random Forest
# ------------------------
print("Feature Importance of the Random Forest Model")
feature_importances = rf_pipeline.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=True)
print("Feature Importance:",importance_df)
print("")


plt.figure(figsize=(8, 5))
plt.barh(X.columns, rf_pipeline.feature_importances_)
plt.title("Random Forest Model Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show(block=True)

# ------------------------
# XGBoost
# ------------------------
print("Feature Importance of the XGBoost Model")
feature_importances = xgb_pipeline.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=True)
print("Feature Importance:",importance_df)


plt.figure(figsize=(8, 5))
plt.barh(X.columns, xgb_pipeline.feature_importances_)
plt.title("XGBRegressor Model Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show(block=True)
print("")
