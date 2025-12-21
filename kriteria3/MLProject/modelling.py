import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load data
X_train = pd.read_csv("healthcareinsurance_preprocessing/X_train.csv")
X_test = pd.read_csv("healthcareinsurance_preprocessing/X_test.csv")
y_train = pd.read_csv("healthcareinsurance_preprocessing/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("healthcareinsurance_preprocessing/y_test.csv").iloc[:, 0]

# Hapus nested=True karena tidak ada parent run
with mlflow.start_run(run_name="insurance_model_training"):
    mlflow.sklearn.autolog()

    # Train model
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics manually
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="insurance_cost_model"
    )

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print("Model training completed successfully!")