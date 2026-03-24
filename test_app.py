from fastapi.testclient import TestClient
from main import app
import time

client = TestClient(app)

print("Loading directory (10% sample)...")
r = client.post("/api/load_directory", json={"directory_path": "/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL", "sample_frac": 0.1})
if r.status_code != 200:
    print("Load failed:", r.text)
    exit(1)
session_id = r.json()["session_id"]
print("Session:", session_id)

print("Training model...")
t_json = {
    "session_id": session_id,
    "algorithm": "xgboost",
    "test_size": 0.2,
    "hyperparams": {"n_estimators": 50, "max_depth": 3}
}
r2 = client.post("/api/train", json=t_json)
if r2.status_code != 200:
    print("Train failed:", r2.text)
    exit(1)

model_id = r2.json()["model_id"]
print("Model:", model_id)

print("Running detection...")
t0 = time.time()
p_json = {
    "session_id": session_id,
    "model_id": model_id,
    "qtta_params": {"base_threshold": 0.5, "alpha": 0.3, "d": 1.0}
}
r3 = client.post("/api/predict", json=p_json)
t1 = time.time()

if r3.status_code != 200:
    print("Predict failed:", r3.text)
else:
    data = r3.json()
    preds = data.get("predictions", [])
    print(f"Success! Got predictions: {len(preds)} in {round(t1-t0, 2)}s")
    if len(preds) > 0:
        print("First prediction:", preds[0])
