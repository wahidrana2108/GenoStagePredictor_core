import os
import glob

from xgboost.testing.data import joblib


# Function to find the latest global model in the "global_model" folder
def find_latest_global_model():
    # Specify the path to the global_model folder
    folder_path = 'global_model'

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # List all files in the folder
    files = glob.glob(os.path.join(folder_path, 'global_model_*.pkl'))

    # Check if there are any model files
    if not files:
        print("No global model found.")
        return None

    # Find the latest model based on the timestamp
    latest_model = max(files, key=os.path.getctime)

    print(f"Latest global model found: {latest_model}")

    return latest_model

# Example usage
latest_global_model_path = find_latest_global_model()
if latest_global_model_path:
    # Load the latest global model
    latest_global_model = joblib.load(latest_global_model_path)
    # Perform any further actions with the latest_global_model
