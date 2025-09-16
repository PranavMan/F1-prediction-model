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
f1_data = pd.read_csv("data/f1_lap_telemetry_2022_2025.csv")

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

# Use all columns except identifiers/targets
feature_cols = [
    c for c in race_level_data.columns 
    if c not in ['year', 'race', 'driver', 'points_first', 'finish_position_first']
]
X = race_level_data[feature_cols]

# ------------------------
# Preprocessing (future-proof small one-hot)
# ------------------------

# Future categorical features (commented for now)
# categorical_features = ["tire_compound", "weather"]
categorical_features = []  # <-- empty list until you add them later

numeric_features = [c for c in feature_cols if c not in categorical_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Only build categorical pipeline if categorical_features exist
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
]) if categorical_features else 'drop'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        # Uncomment this once you add tire/weather columns
        # ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# ------------------------
# Print Out Header
# ------------------------
print("This is an F1 points prediction model using Decision Tree, Random Forest, and XGBoost\n")
print("This model uses data from the F1 2022 season, future updates will include 2023, 2024, and 2025 seasons\n")
print("This model outputs the first 10 predictions from each model, the actual values, the validation MAE, and the mean cross-validated MAE\n")
print("Then the model displays the feature importance for each model\n")
print("--------------------------------------------------------------\n")

# ------------------------
# Train-test split
# ------------------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=1)

# ------------------------
# Decision Tree
# ------------------------
print("Model 1: Decision Tree Regressor\n")
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
print("Model 2: Random Forest Regressor\n")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=1))
])
rf_pipeline.fit(X_train, y_train)
rf_model = rf_pipeline.named_steps['model']
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
print("Model 3: XGBoost Regressor\n")
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(enable_categorical=False, random_state=42))
])
xgb_pipeline.fit(X_train, y_train)
xgb_model = xgb_pipeline.named_steps['model']
xgb_preds = np.round(xgb_pipeline.predict(X_valid)).astype(int)
xgb_mae = mean_absolute_error(y_valid, xgb_preds)
print("XGBoost Model Prediction vs Actual Values & Validation MAE & CV Scores")
print("Predictions:", xgb_preds[:10])
print("Actual:", y_valid[:10].values)
print("MAE:", xgb_mae)
print("Mean CV MAE:", -cross_val_score(xgb_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error').mean())
print()



def get_transformed_feature_names(pipeline):
    """
    Return the column names after the ColumnTransformer inside your pipeline.
    Handles numeric passthrough (with imputer) and categorical OneHotEncoder.
    """
    pre = pipeline.named_steps['preprocessor']

    # Helper to fetch the original column list for a given transformer name
    def cols_for(name):
        for n, trans, cols in pre.transformers_:
            if n == name:
                # 'cols' can be a list-like or an array; normalize to list
                return list(cols)
        return []

    num_cols = cols_for('num')
    cat_cols = cols_for('cat')

    # Start with numeric columns (imputer keeps names)
    names = list(num_cols)

    # Add OHE-expanded categorical names if any categorical columns exist
    cat_names = []
    if cat_cols:
        ohe = pre.named_transformers_['cat'].named_steps['encoder']
        # get_feature_names_out returns names like "<original_col>_<category>"
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()

    names.extend(cat_names)
    return names

# ==============================
# Feature importance (works with pipelines)
# ==============================
print("-------------------------------")
print("Feature Importance for Each Model")
print("-------------------------------\n")

def plot_feature_importance(pipeline, model_name, feature_cols):
    """Plot sorted feature importances for a model inside a pipeline."""
    # Get trained estimator from pipeline
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Feature importances
    importances = model.feature_importances_

    # Future-proof: get real feature names (handles one-hot automatically)
    try:
        feature_names = preprocessor.get_feature_names_out(feature_cols)
    except:
        feature_names = feature_cols  # fallback when no categorical features

    # Build dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=True)

    # Print for inspection
    print(f"\nFeature Importance of the {model_name} Model")
    print(importance_df)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title(f"{model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show(block=True)

# Call for each fitted pipeline
plot_feature_importance(dt_pipeline, "Decision Tree", feature_cols)
plot_feature_importance(rf_pipeline, "Random Forest", feature_cols)
plot_feature_importance(xgb_pipeline, "XGBoost", feature_cols)


#Look into K-fold cross validation for splitting data