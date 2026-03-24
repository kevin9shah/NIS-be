import pandas as pd
import numpy as np

df = pd.read_csv('/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv', nrows=5000)
for col in df.columns:
    if df[col].dtype == object or str(df[col].dtype) == 'category':
        print(f"STRING/CAT COL: {col} -> example: {df[col].iloc[0]}")
    elif pd.api.types.is_numeric_dtype(df[col]):
        if df[col].nunique() < 10:
            print(f"NUMERIC CAT COL: {col} -> unique vals: {df[col].nunique()}")
