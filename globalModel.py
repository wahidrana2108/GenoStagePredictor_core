import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
import os
import shutil
from datetime import datetime
import time

from localMachine import X_train, y_train, X_test, y_test


# Define the global XGBoost model wrapper
class GlobalXGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier()

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# Initialize the global XGBoost model
global_model = GlobalXGBoostModel()

# Create the "global_model" folder if it does not exist
os.makedirs('global_model', exist_ok=True)
os.makedirs('all_global_models', exist_ok=True)  # Create folder for all global models

# Continuous loop to check for new local models
while True:
    # Check for local model files in the "local_models" folder
    local_models_folder = 'local_models'
    local_model_files = [os.path.join(local_models_folder, f) for f in os.listdir(local_models_folder) if f.endswith('.pkl')]

    if local_model_files:
        print(f"Found {len(local_model_files)} local models. Updating the global model.")

        # Aggregate local model updates (simulate aggregation from local servers)
        for local_model_filename in local_model_files:
            local_xgb_model = joblib.load(local_model_filename)

            # Simulate model updates by taking an average of hyperparameters
            global_params = global_model.get_params()
            local_params = local_xgb_model.get_params()

            averaged_params = {}
            for param_name in global_params.keys():
                if isinstance(global_params[param_name], (int, float)):
                    averaged_params[param_name] = (global_params[param_name] + local_params[param_name]) / 2.0
                else:
                    averaged_params[param_name] = global_params[param_name]

            # Update global model with averaged parameters
            global_model.set_params(**averaged_params)

            # Move the processed local model to the "trainer" folder
            trainer_folder = 'trainer'
            os.makedirs(trainer_folder, exist_ok=True)
            destination_path = os.path.join(trainer_folder, os.path.basename(local_model_filename))
            shutil.move(local_model_filename, destination_path)

        # Perform global training with the aggregated model
        global_model.fit(X_train, y_train)

        # Evaluate the accuracy after updating the global model
        y_pred_global = global_model.predict(X_test)
        accuracy_global = accuracy_score(
            y_test, y_pred_global)
        print(f"Global Model Accuracy: {accuracy_global}")

        # Save the trained global XGBoost model to a file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        global_model_filename = os.path.join('all_global_models', f'global_model_{timestamp}.pkl')
        joblib.dump(global_model, global_model_filename)

        # Move the latest global model to the "global_model" folder
        latest_global_model_filename = os.path.join('global_model', f'global_model_latest.pkl')
        shutil.copy(global_model_filename, latest_global_model_filename)

        print("Global model updated.")

    # Add a delay to avoid continuous checking
    time.sleep(10)  # Adjust the delay as needed
