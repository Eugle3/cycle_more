"""
Shared helpers for loading data/models and running recommendations.

This keeps Streamlit/UI concerns separate so the same logic can be used
from FastAPI (and any other client).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import pandas as pd

from .llm_distance import change_route_distance as _change_route_distance
from .llm_distance import run_distance_change_query as _run_distance_change_query
from .recommender import recommend_similar_routes

BASE_DIR = Path(__file__).resolve().parent

_data_cache: Tuple[pd.DataFrame, List[str]] | None = None
_model_cache: Tuple[Any, Any] | None = None


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load the engineered routes and return (df, feature_cols)."""
    global _data_cache
    if _data_cache is None:
        df = pd.read_csv(BASE_DIR / "Data_Engineered.csv")
        feature_cols = df.drop(["id", "name"], axis=1).columns.tolist()
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
