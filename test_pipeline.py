import pandas as pd
from core.data_loader import load_directory_sampled
from core.preprocessor import NetworkPreprocessor
from core.models import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)

print("Loading 5% sample directly via backend loader...")
df = load_directory_sampled('/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL', sample_frac=0.05)

df = df.rename(columns={"Attack_label": "label"})
if "Attack_type" in df.columns:
    df = df.drop(columns=["Attack_type"])

print("Applying Upload Coercion...")
for col in df.columns:
    if col != "label":
        if df[col].dtype == object or str(df[col].dtype) == 'category':
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.isna().sum() > len(df) * 0.5:
                df[col] = pd.factorize(df[col].astype(str))[0]
            else:
                df[col] = coerced

df = df.dropna(axis=1, how='all')
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

X = df.drop(columns=["label"])
y = df["label"].values

print(f"X shape: {X.shape}")
print(f"y class dist: {(y==0).sum()} normal, {(y==1).sum()} anomaly")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = NetworkPreprocessor()
trainer = ModelTrainer()

X_train_scaled, feature_names = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

print("Training XGBoost...")
trainer.train("xgboost", X_train_scaled, y_train)

metrics = trainer.evaluate(X_test_scaled, y_test)
print(f"Metrics: {metrics}")
