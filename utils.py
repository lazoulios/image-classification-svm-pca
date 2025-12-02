import os
import joblib

MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"[utils] Folder Created '{MODEL_DIR}'.")

def save_model(model, filename):
    filepath = os.path.join(MODEL_DIR, filename)
    print(f"[utils] Saving: {filepath}...")
    joblib.dump(model, filepath)
    print("[utils] Save completed.")

def load_model(filename):
    """Loads a model from the models subfolder if it exists."""
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        print(f"[utils] Loading: {filepath}...")
        return joblib.load(filepath)
    else:
        return None