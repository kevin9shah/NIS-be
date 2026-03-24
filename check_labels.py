import pandas as pd

print("Reading full CSV...")
df = pd.read_csv('/Users/kabra_jay/Downloads/archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv', usecols=["Attack_label"], low_memory=False)

print("Full dataset label distribution:")
print(df["Attack_label"].value_counts(dropna=False))

df_sampled = df.sample(frac=0.5, random_state=42)
print("Sampled 0.5 label distribution:")
print(df_sampled["Attack_label"].value_counts(dropna=False))
