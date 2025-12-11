#!/usr/bin/env python3
"""
Quick script to add cluster column to Data_Engineered.csv
"""
import pandas as pd
import joblib
from pathlib import Path

# Load data
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "Data_Engineered.csv")

# Check if cluster already exists
if "cluster" in df.columns:
    print("✓ Cluster column already exists!")
    print(f"  Cluster distribution:\n{df['cluster'].value_counts().sort_index()}")
    exit(0)

# Load models
print("Loading models...")
kmeans = joblib.load(BASE_DIR / "kmeans.pkl")
scaler = joblib.load(BASE_DIR / "KNN_scaler.pkl")

# Get feature columns (exclude metadata)
non_feature_cols = ["id", "name", "region"]
feature_cols = df.drop(non_feature_cols, axis=1).columns.tolist()

print(f"Using {len(feature_cols)} features for clustering")

# Scale features
X = df[feature_cols]
X_scaled = scaler.transform(X)

# Predict clusters
print("Predicting clusters...")
clusters = kmeans.predict(X_scaled)

# Add cluster column
df["cluster"] = clusters

# Save updated CSV
output_path = BASE_DIR / "Data_Engineered.csv"
print(f"Saving to {output_path}...")
df.to_csv(output_path, index=False)

print(f"✓ Done! Added cluster column with {len(set(clusters))} clusters")
print(f"  Cluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}")
