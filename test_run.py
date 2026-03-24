import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from core.data_loader import load_directory_sampled
from core.preprocessor import NetworkPreprocessor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Loading data...")
df = load_directory_sampled('/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL', sample_frac=0.1)

print(f"Loaded {len(df)} rows.")

df = df.rename(columns={"Attack_label": "label"})
if "Attack_type" in df.columns:
    df = df.drop(columns=["Attack_type"])

# emulate upload.py
for col in df.columns:
    if col != "label":
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='ignore')
            if converted.dtype == object:
                df[col] = pd.factorize(converted.astype(str))[0]
            else:
                df[col] = converted

df = df.dropna(axis=1, how='all')

for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

print("Label distribution:")
print(df["label"].value_counts())

X = df.drop(columns=["label"])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Preprocessing...")
prep = NetworkPreprocessor()
X_train_scaled, fnames = prep.fit_transform(X_train)
X_test_scaled = prep.transform(X_test)

print(f"Features: {len(fnames)}")

n_neg = int(np.sum(y_train == 0))
n_pos = int(np.sum(y_train == 1))
spw = n_neg / n_pos if n_pos > 0 else 1.0

print(f"Training XGBoost WITH scale_pos_weight={spw:.2f}")
m1 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, scale_pos_weight=spw, eval_metric="logloss")
m1.fit(X_train_scaled, y_train)
p1 = m1.predict(X_test_scaled)
print(f"WITH SPW -> Acc: {accuracy_score(y_test, p1):.4f}, F1: {f1_score(y_test, p1):.4f}, CM: {confusion_matrix(y_test, p1).ravel()}")

print(f"Training XGBoost WITHOUT scale_pos_weight")
m2 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, eval_metric="logloss")
m2.fit(X_train_scaled, y_train)
p2 = m2.predict(X_test_scaled)
print(f"WITHOUT SPW -> Acc: {accuracy_score(y_test, p2):.4f}, F1: {f1_score(y_test, p2):.4f}, CM: {confusion_matrix(y_test, p2).ravel()}")
