import pandas as pd
from core.data_loader import load_directory_sampled
from core.preprocessor import NetworkPreprocessor
import numpy as np

print("Loading data...")
df = load_directory_sampled('/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL', sample_frac=0.01)

df = df.rename(columns={"Attack_label": "label"})
if "Attack_type" in df.columns:
    df = df.drop(columns=["Attack_type"])

for col in df.columns:
    if col != "label":
        if df[col].dtype == object or str(df[col].dtype) == 'category':
            coerced = pd.to_numeric(df[col], errors='coerce')
            if coerced.isna().sum() > len(df) * 0.5:
                df[col] = pd.factorize(df[col].astype(str))[0]
            else:
                df[col] = coerced

df = df.dropna(axis=1, how='all')

preprocessor = NetworkPreprocessor()
try:
    processed_df, feature_names = preprocessor.fit_transform(df)
    print("Success! No NaNs.")
except ValueError as e:
    print("Caught ValueError:", e)

# Let's inspect BEFORE PCA
df2 = df.copy()

useless_ids = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.hw_mac", "arp.dst.hw_mac"]
df2 = df2.drop(columns=[c for c in useless_ids if c in df2.columns], errors='ignore')

# Use all numeric columns
keep_cols = df2.select_dtypes(include=['number']).columns.tolist()
df2 = df2[keep_cols].copy()

# Step 1-3
df2 = preprocessor._impute_missing(df2, fit=True)
df2 = preprocessor._clip_outliers(df2)
df2 = preprocessor._engineer_features(df2)

print("\n--- Inspecting pre-scaler NaNs ---")
nan_cols = df2.columns[df2.isna().sum() > 0].tolist()
print("Columns with NaN after all cleaning:", nan_cols)
for c in nan_cols:
    print(f"  {c}: {df2[c].isna().sum()} NaNs")

df2 = df2.replace([np.inf, -np.inf], np.nan)
inf_cols = df2.columns[df2.isna().sum() > 0].tolist()
print("Columns with Inf/NaN before scaler:", inf_cols)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
try:
    scaled = scaler.fit_transform(df2)
    print("Scaler succeeded.")
    
    scaled_df = pd.DataFrame(scaled)
    scaled_nan = scaled_df.columns[scaled_df.isna().sum() > 0].tolist()
    print("Columns with NaN AFTER scaler:", scaled_nan)
except Exception as e:
    print("Scaler failed:", e)
