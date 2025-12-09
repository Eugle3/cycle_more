"""
Shared helpers for loading data/models and running recommendations.

This keeps Streamlit/UI concerns separate so the same logic can be used
from FastAPI (and any other client).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import pandas as pd

from .llm_distance import change_route_distance as _change_route_distance
from .llm_distance import run_distance_change_query as _run_distance_change_query
from .recommender import recommend_similar_routes

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


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load the engineered routes and return (df, feature_cols)."""
    global _data_cache
    if _data_cache is None:
        df = pd.read_csv(BASE_DIR / "Data_Engineered.csv")
        # Drop non-feature columns (id, name, region)
        feature_cols = df.drop(["id", "name", "region"], axis=1).columns.tolist()
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
