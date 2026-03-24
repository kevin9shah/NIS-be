import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset natively...")
df = pd.read_csv('/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv', nrows=500000, low_memory=False)

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

print("Dropping useless IDs...")
useless_ids = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.hw_mac", "arp.dst.hw_mac"]
df = df.drop(columns=[c for c in useless_ids if c in df.columns], errors='ignore')

print("Dropping completely NaN...")
df = df.dropna(axis=1, how='all')

print("Imputing...")
df = df.fillna(0)

# boolean
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

X = df.drop(columns=["label"])
y = df["label"].astype(int).values

print(f"X shape: {X.shape}")
print(pd.Series(y).value_counts())

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Training XGBoost...")
m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
m.fit(X_train, y_train)

p = m.predict(X_test)
print("Accuracy:", accuracy_score(y_test, p))
print(classification_report(y_test, p))

print("Top 15 Feature Importances:")
importances = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(15))
