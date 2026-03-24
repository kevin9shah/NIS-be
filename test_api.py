import requests
import time

base = 'http://localhost:8000/api'

print("Loading directory...")
r = requests.post(f"{base}/load_directory", json={"directory_path": "/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL", "sample_frac": 0.05})
session_id = r.json()["session_id"]
print("Session:", session_id)

print("Training model...")
t_json = {
    "session_id": session_id,
    "algorithm": "xgboost",
    "test_size": 0.2,
    "hyperparams": {"n_estimators": 50, "max_depth": 3}
}
r2 = requests.post(f"{base}/train", json=t_json)
model_id = r2.json().get("model_id")
print("Model:", model_id)

if not model_id:
    print("Train failed:", r2.text)
    exit(1)

print("Running detection...")
p_json = {
    "session_id": session_id,
    "model_id": model_id,
    "qtta_params": {"base_threshold": 0.5, "alpha": 0.3, "d": 1.0}
}
r3 = requests.post(f"{base}/predict", json=p_json)
if r3.status_code != 200:
    print("Predict failed:", r3.text)
else:
    print("Success! Got predictions:", len(r3.json().get("predictions", [])))
