import pandas as pd
from typing import Any, Dict, List, Union


def recommend_similar_routes(
    input_features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model,
    scaler,
    df: pd.DataFrame,
    feature_cols: List[str],
    n_recommendations: int = 5,
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

    # Scale + find neighbors
    input_scaled = scaler.transform(input_df)
    distances, indices = model.kneighbors(
        input_scaled, n_neighbors=min(n_recommendations, len(df))
    )

    recommendations = []
    for dist, idx in zip(distances[0], indices[0]):
        rec = df.iloc[idx]
        recommendations.append(
            {
                "route_id": int(rec["id"]),
                "route_name": str(rec["name"]),
                "distance_m": float(rec["distance_m"]),
                "ascent_m": float(rec["ascent_m"]),
                "duration_s": float(rec["duration_s"]),
                "turn_density": float(rec["Turn_Density"]),
                "similarity_score": float(dist),
            }
        )

    return recommendations
