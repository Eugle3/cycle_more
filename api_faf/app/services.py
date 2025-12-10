"""
Shared helpers for loading data/models and running recommendations.

This keeps Streamlit/UI concerns separate so the same logic can be used
from FastAPI (and any other client).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import random

import joblib
import pandas as pd
import numpy as np

from .llm_distance import change_route_distance as _change_route_distance
from .llm_distance import run_distance_change_query as _run_distance_change_query
from .recommender import recommend_similar_routes
from .cluster_labels import CLUSTER_LABELS

# Add FAF module to path for GPX processing
# In Docker: services.py is at /app/app/services.py, FAF is at /app/FAF/
# In local: services.py is at cycle_more/api_faf/app/services.py, FAF is at cycle_more/FAF/
FAF_PATH = Path(__file__).resolve().parent.parent.parent / "FAF"
# Check if running in Docker (FAF would be 2 levels up instead of 3)
if not FAF_PATH.exists():
    FAF_PATH = Path(__file__).resolve().parent.parent / "FAF"
if str(FAF_PATH) not in sys.path:
    sys.path.insert(0, str(FAF_PATH))

from gpx_to_features import process_gpx_file

BASE_DIR = Path(__file__).resolve().parent

_data_cache: Tuple[pd.DataFrame, List[str]] | None = None
_model_cache: Tuple[Any, Any] | None = None
_kmeans_cache: Any | None = None


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load the engineered routes and return (df, feature_cols)."""
    global _data_cache
    if _data_cache is None:
        df = pd.read_csv(BASE_DIR / "Data_Engineered.csv")
        # Drop non-feature columns (id, name, region, cluster)
        # Note: cluster is used for curveball filtering but not as a model feature
        non_feature_cols = ["id", "name", "region"]
        if "cluster" in df.columns:
            non_feature_cols.append("cluster")
        feature_cols = df.drop(non_feature_cols, axis=1).columns.tolist()
        _data_cache = (df, feature_cols)
    return _data_cache


def load_model_and_scaler() -> Tuple[Any, Any]:
    """Load the trained model and scaler used for similarity search."""
    global _model_cache
    if _model_cache is None:
        model = joblib.load(BASE_DIR / "model.pkl")
        scaler = joblib.load(BASE_DIR / "scaler.pkl")
        _model_cache = (model, scaler)
    return _model_cache


def load_kmeans() -> Any:
    """Load the trained k-means clustering model."""
    global _kmeans_cache
    if _kmeans_cache is None:
        _kmeans_cache = joblib.load(BASE_DIR / "kmeans.pkl")
    return _kmeans_cache


def predict_cluster(
    input_features: Union[Dict[str, Any], pd.Series, pd.DataFrame]
) -> Tuple[int, str]:
    """
    Predict which cluster a route belongs to.

    Args:
        input_features: Route features (same format as recommend_routes)

    Returns:
        Tuple of (cluster_id, cluster_label)
    """
    # Load models
    kmeans = load_kmeans()
    _, scaler = load_model_and_scaler()
    _, feature_cols = load_data()

    # Normalize input into a single-row DataFrame
    if isinstance(input_features, dict):
        input_df = pd.DataFrame([input_features])
    elif isinstance(input_features, pd.Series):
        input_df = input_features.to_frame().T
    else:
        input_df = input_features.copy()

    # Align columns
    input_df = input_df[feature_cols]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict cluster
    cluster_id = int(kmeans.predict(input_scaled)[0])
    cluster_label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")

    return cluster_id, cluster_label


def recommend_routes(
    input_features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    n_recommendations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Recommend similar routes given a feature payload.

    Expects the model/scaler/data to be on disk next to this module.
    """
    df, feature_cols = load_data()
    model, scaler = load_model_and_scaler()
    return recommend_similar_routes(
        input_features=input_features,
        model=model,
        scaler=scaler,
        df=df,
        feature_cols=feature_cols,
        n_recommendations=n_recommendations,
    )


def change_route_distance(df: pd.DataFrame, route_id: str, multiplier: float):
    """Thin wrapper around the LLM/fallback distance change helper."""
    return _change_route_distance(df, route_id, multiplier)


def run_distance_change_query(df: pd.DataFrame, user_question: str):
    """
    Ask Gemini to choose a route_id + multiplier. Returns (route_df, tool_args, err).
    """
    return _run_distance_change_query(df, user_question)


def process_gpx_upload(gpx_content: bytes, n_recommendations: int = 5) -> List[Dict[str, Any]]:
    """
    Process a GPX file upload and return route recommendations.

    This is the complete pipeline for GPX uploads:
    1. Parse GPX and extract coordinates (smart sampling to 70 waypoints)
    2. Call ORS API to get route features
    3. Process and engineer features (same as training data)
    4. Use existing KNN model to find similar routes

    Args:
        gpx_content: Bytes content of the GPX file
        n_recommendations: Number of recommendations to return (default 5)

    Returns:
        List of route recommendations with similarity scores

    Raises:
        ValueError: If GPX parsing fails
        Exception: If processing or model prediction fails
    """
    # Step 1-5: GPX → engineered features (uses FAF modules)
    engineered_features = process_gpx_file(gpx_content)

    # Step 6: Get recommendations using existing model/scaler
    recommendations = recommend_routes(engineered_features, n_recommendations)

    return recommendations


def recommend_with_curveball(
    input_features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    n_similar: int = 5,
) -> Dict[str, Any]:
    """
    Get route recommendations including a "curveball" from a different cluster.

    Returns n_similar KNN recommendations plus 1 curveball: the nearest neighbor
    from a different cluster than the user's route.

    Args:
        input_features: Route features (same format as recommend_routes)
        n_similar: Number of similar routes to return (default 5)

    Returns:
        Dict with:
            - "similar": List of n_similar KNN recommendations
            - "curveball": Single nearest route from a different cluster
            - "user_cluster_id": User's predicted cluster ID
            - "user_cluster_label": User's cluster label
            - "curveball_cluster_id": Curveball's cluster ID
            - "curveball_cluster_label": Curveball's cluster label
    """
    # Get user's cluster
    user_cluster_id, user_cluster_label = predict_cluster(input_features)

    # Get similar routes (KNN)
    similar_routes = recommend_routes(input_features, n_recommendations=n_similar)

    # Get curveball from different cluster
    df, feature_cols = load_data()
    model, scaler = load_model_and_scaler()

    # Normalize input
    if isinstance(input_features, dict):
        input_df = pd.DataFrame([input_features])
    elif isinstance(input_features, pd.Series):
        input_df = input_features.to_frame().T
    else:
        input_df = input_features.copy()

    input_df = input_df[feature_cols]
    input_scaled = scaler.transform(input_df)

    # Filter to routes in different clusters
    different_cluster_routes = df[df['cluster'] != user_cluster_id]

    if len(different_cluster_routes) == 0:
        # Edge case: only one cluster exists (shouldn't happen with 10 clusters)
        curveball_route = None
        curveball_cluster_id = user_cluster_id
        curveball_cluster_label = user_cluster_label
    else:
        # Get features for different cluster routes
        different_X = different_cluster_routes[feature_cols]
        different_X_scaled = scaler.transform(different_X)

        # Find nearest neighbor from different clusters
        distances, indices = model.kneighbors(input_scaled, n_neighbors=len(different_cluster_routes))

        # Find the first route that's in a different cluster
        for dist, idx in zip(distances[0], indices[0]):
            route = df.iloc[idx]
            if route['cluster'] != user_cluster_id:
                curveball_cluster_id = int(route['cluster'])
                curveball_cluster_label = CLUSTER_LABELS.get(
                    curveball_cluster_id, f"Cluster {curveball_cluster_id}"
                )

                curveball_route = {
                    "route_id": int(route["id"]),
                    "route_name": str(route["name"]),
                    "distance_m": float(route["distance_m"]),
                    "ascent_m": float(route["ascent_m"]),
                    "duration_s": float(route["duration_s"]),
                    "turn_density": float(route["Turn_Density"]),
                    "similarity_score": float(dist),
                    "primary_surface": _get_primary_surface(route),
                }
                break

    return {
        "similar": similar_routes,
        "curveball": curveball_route,
        "user_cluster_id": user_cluster_id,
        "user_cluster_label": user_cluster_label,
        "curveball_cluster_id": curveball_cluster_id,
        "curveball_cluster_label": curveball_cluster_label,
    }


def _get_primary_surface(row: pd.Series) -> str:
    """Helper to determine primary surface type (copied from recommender.py)."""
    from .recommender import get_primary_surface
    return get_primary_surface(row)
