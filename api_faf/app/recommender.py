import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union

from .route_namer import enhance_route_name


def get_primary_surface(row: pd.Series) -> str:
    """
    Determine the primary surface type from route features.

    Returns the surface type with the highest percentage.
    """
    surface_columns = {
        'Gravel_Tracks': 'Gravel',
        'Paved_Paths': 'Paved Path',
        'Paved_Road': 'Paved Road',
        'Cycle Track': 'Cycle Track',
        'Main Road': 'Main Road',
        'Pedestrian': 'Pedestrian',
        'Unknown_Way': 'Unknown',
        'Unknown Surface': 'Unknown',
        'Other': 'Other',
    }

    # Find which surface has the highest value
    max_surface = None
    max_value = 0.0

    for col, display_name in surface_columns.items():
        if col in row.index:
            value = float(row.get(col, 0.0))
            if value > max_value:
                max_value = value
                max_surface = display_name

    # If no clear surface found, check on_road vs off_road
    if not max_surface or max_value < 0.1:
        if 'on_road' in row.index and float(row.get('on_road', 0)) > 0.5:
            return 'On-Road'
        elif 'off_road' in row.index and float(row.get('off_road', 0)) > 0.5:
            return 'Off-Road'
        return 'Mixed'

    return max_surface


def recommend_similar_routes(
    input_features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model,
    scaler,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_recommendations: int = 5,
    surface_weight: float = 2.0,
):
    """
    Minimal version of the notebook helper to run inside the API.

    Parameters
    ----------
    input_features : dict | Series | DataFrame
        Single row of features expected by the model.
    model : fitted NearestNeighbors (or similar)
        The trained model loaded in your API (e.g., joblib.load).
    scaler : fitted transformer
        The same scaler/column transformer used during training.
    df : DataFrame
        Original routes dataframe with 'id' and 'name' columns for lookup.
    feature_cols : list[str]
        Columns used to train the model (order matters).
    n_recommendations : int
        Number of similar routes to return.
    surface_weight : float
        Multiplier for surface/way type features to increase their importance.
        Default 2.0 means surface features are 2x more important than other features.
        Set to 1.0 for no weighting.
    """
    # Normalize input into a single-row DataFrame
    if isinstance(input_features, dict):
        input_df = pd.DataFrame([input_features])
    elif isinstance(input_features, pd.Series):
        input_df = input_features.to_frame().T
    else:
        input_df = input_features.copy()

    # Validate and align columns
    missing = [col for col in feature_cols if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    input_df = input_df[feature_cols]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Apply surface feature weighting if requested
    if surface_weight != 1.0:
        # Surface/way features are MinMaxScaled and appear at indices 7-18:
        # Cycleway, on_road, off_road, Gravel_Tracks, Paved_Paths, Other,
        # Unknown Surface, Paved_Road, Pedestrian, Unknown_Way, Cycle Track, Main Road
        input_weighted = input_scaled.copy()
        input_weighted[:, 7:19] *= surface_weight  # Multiply surface features by weight
    else:
        input_weighted = input_scaled

    # Find neighbors using weighted features
    distances, indices = model.kneighbors(
        input_weighted, n_neighbors=min(n_recommendations, len(df))
    )

    recommendations = []
    for dist, idx in zip(distances[0], indices[0]):
        rec = df.iloc[idx]
        route_id = int(rec["id"])
        route_name = str(rec["name"])
        distance_m = float(rec["distance_m"])
        ascent_m = float(rec["ascent_m"])

        # Enhance generic route names using geocoding (with fallback to distance/ascent)
        route_name = enhance_route_name(
            route_id, route_name,
            distance_m=distance_m,
            ascent_m=ascent_m
        )

        recommendations.append(
            {
                "route_id": route_id,
                "route_name": route_name,
                "distance_m": distance_m,
                "ascent_m": ascent_m,
                "duration_s": float(rec["duration_s"]),
                "turn_density": float(rec["Turn_Density"]),
                "similarity_score": float(dist),
                "primary_surface": get_primary_surface(rec),
            }
        )

    return recommendations
