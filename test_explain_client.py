from fastapi.testclient import TestClient
from main import app
import logging
logging.basicConfig(level=logging.INFO)

client = TestClient(app)

print("Loading directory...")
r = client.post("/api/load_directory", json={"directory_path": "/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL", "sample_frac": 0.05})
session_id = r.json()["session_id"]
print("Session:", session_id)

print("Training model...")
t_json = {
    "session_id": session_id,
    "algorithm": "xgboost",
    "test_size": 0.2,
    "hyperparams": {"n_estimators": 10, "max_depth": 3}
}
r2 = client.post("/api/train", json=t_json)
model_id = r2.json()["model_id"]
print("Model:", model_id)

print("Running explain...")
p_json = {
    "session_id": session_id,
    "model_id": model_id,
}
r3 = client.post("/api/explain", json=p_json)
if r3.status_code != 200:
    print("Explain failed:", r3.text)
else:
    data = r3.json()
    print("Success!")
    print("Features:", data.get("feature_names"))
    print("SHAP top:", data.get("shap_importance")[:3])
    print("PDP generated:", len(data.get("pdp")))
