"""
Retrain KNN model on updated dataset with 24,766 routes.
This script replicates the key cells from the KNN training notebook.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import joblib
import sys

print("=" * 80)
print("RETRAINING KNN MODEL")
print("=" * 80)

# 1. Load data
print("\n1. Loading data...")
df = pd.read_csv("api_faf/app/Data_Engineered.csv")
print(f"   ✅ Loaded {len(df):,} routes")
print(f"   Columns: {df.shape[1]}")

# 2. Prepare features
print("\n2. Preparing features...")
X = df.drop(['id', 'name', 'region', 'cluster'], axis=1)
print(f"   ✅ Feature matrix created: {X.shape}")

# 3. Scale features
print("\n3. Scaling features...")
scaler = ColumnTransformer(transformers=[
    ('standard', StandardScaler(), [
        'distance_m', 'duration_s', 'ascent_m', 'descent_m',
        'Turn_Density', 'steps', 'turns'
    ]),
    ('minmax', MinMaxScaler(), [
        'Cycleway', 'on_road', 'off_road', 'Gravel_Tracks', 'Paved_Paths',
        'Other', 'Unknown Surface', 'Paved_Road', 'Pedestrian', 'Unknown_Way',
        'Cycle Track', 'Main Road', 'Steep Section', 'Moderate Section',
        'Flat Section', 'Downhill Section', 'Steep Downhill Section'
    ]),
], remainder='passthrough')

X_scaled = scaler.fit_transform(X)
print(f"   ✅ Features scaled: {X_scaled.shape}")

# 4. Train KNN model
print("\n4. Training KNN model...")
knn_optimal = NearestNeighbors(n_neighbors=5, metric='cosine')
knn_optimal.fit(X_scaled)

print(f"   ✅ KNN model trained!")
print(f"      k = {knn_optimal.n_neighbors}")
print(f"      metric = '{knn_optimal.metric}'")
print(f"      training samples = {X_scaled.shape[0]:,}")

# 5. Save models
print("\n5. Saving models...")
output_dir = "Notebooks/2.)KNN Model/"
joblib.dump(knn_optimal, f"{output_dir}model.pkl")
joblib.dump(scaler, f"{output_dir}scaler.pkl")

print(f"   ✅ Saved to {output_dir}")
print(f"      - model.pkl")
print(f"      - scaler.pkl")

print("\n" + "=" * 80)
print("✅ RETRAINING COMPLETE!")
print("=" * 80)
print(f"\nNext step: Copy files to api_faf/app/")
print(f"  cp '{output_dir}model.pkl' api_faf/app/model.pkl")
print(f"  cp '{output_dir}scaler.pkl' api_faf/app/scaler.pkl")
