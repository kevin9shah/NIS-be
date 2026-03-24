import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("Reading CSV...")
df = pd.read_csv('/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv', nrows=200000, low_memory=False)

print("Mapping labels...")
df = df.rename(columns={"Attack_label": "label"})
if "Attack_type" in df.columns:
    df = df.drop(columns=["Attack_type"])

print("Dropping useless...")
useless = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.hw_mac", "arp.dst.hw_mac", "arp.src.proto_ipv4", "arp.dst.proto_ipv4"]
df = df.drop(columns=[c for c in useless if c in df.columns], errors='ignore')

print("Factorizing strings...")
for col in df.columns:
    if col != "label" and df[col].dtype == object:
        df[col] = pd.factorize(df[col].astype(str))[0]

print("Dropping NaNs...")
df = df.dropna(axis=1, how='all')
df = df.fillna(0)

X = df.drop(columns=["label"])
y = df["label"].astype(int).values

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Training default XGBoost...")
m = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
m.fit(X_train, y_train)

p = m.predict(X_test)
print(confusion_matrix(y_test, p))
print(classification_report(y_test, p))
