"""
Data processing functions for converting raw ORS data to percentages
and applying feature engineering.

These functions replicate the notebook logic from:
- 1.a)Data Processing/2).Data_processing_Rob_day1.ipynb
- 1.a)Data Processing/3).Feature Engineering.ipynb
"""

import ast
from typing import Dict, Any


# Mappings from ORS codes to category names
SURFACE_MAP = {
    0: "Unknown",
    1: "Paved",
    2: "Unpaved",
    3: "Asphalt",
    4: "Concrete",
    5: "Cobblestone",
    6: "Metal",
    7: "Wood",
    8: "Compacted Gravel",
    9: "Fine Gravel",
    10: "Gravel",
    11: "Dirt",
    12: "Ground",
    13: "Ice",
    14: "Paving Stones",
    15: "Sand",
    16: "Woodchips",
    17: "Grass",
    18: "Grass Paver"
}

WAYTYPE_MAP = {
    0: "Unknown",
    1: "State Road",
    2: "Road",
    3: "Street",
    4: "Path",
    5: "Track",
    6: "Cycleway",
    7: "Footway",
    8: "Steps",
    9: "Ferry",
    10: "Construction"
}

STEEPNESS_MAP = {
    -5: "downhill_extreme (<-15%)",
    -4: "downhill_very_steep (-15% to -10%)",
    -3: "downhill_steep (-10% to -7%)",
    -2: "downhill_moderate (-7% to -5%)",
    -1: "downhill_gentle (-5% to 0%)",
     0: "flat (0%)",
     1: "uphill_gentle (0% to 3%)",
     2: "uphill_moderate (3% to 5%)",
     3: "uphill_steep (5% to 7%)",
     4: "uphill_very_steep (7% to 10%)",
     5: "uphill_extreme (>10%)"
}


def parse_segment_percentages(data, mapping: Dict[int, str]) -> Dict[str, float]:
    """
    Convert ORS segment data [[start, end, code], ...] to percentage distribution.

    Args:
        data: List of [start, end, code] segments from ORS, or string representation
        mapping: Dict mapping codes to category names

    Returns:
        Dict of {category: percentage}
    """
    # Convert string to list if needed (for CSV data)
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except:
            return {}

    if not data:
        return {}

    totals = {}
    total_len = 0

    for seg in data:
        start, end, code = seg
        seg_len = end - start
        total_len += seg_len

        # Map code to name
        name = mapping.get(code, f"Unknown_{code}")
        totals[name] = totals.get(name, 0) + seg_len

    # Convert to percentages
    if total_len == 0:
        return {}

    return {
        name: round(length / total_len * 100, 2)
        for name, length in totals.items()
    }


def process_single_route(raw_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single route's raw ORS features into percentage-based features.

    Replicates the logic from Data_processing_Rob_day1.ipynb for a single route.

    Args:
        raw_features: Dict with raw ORS data (from extract_single_route_features)

    Returns:
        Dict with processed features including surface/waytype/steepness percentages
    """
    processed = {
        "id": raw_features.get("id", "uploaded"),
        "distance_m": raw_features["distance_m"],
        "duration_s": raw_features["duration_s"],
        "ascent_m": raw_features["ascent_m"],
        "descent_m": raw_features["descent_m"],
        "steps": raw_features["steps"],
        "turns": raw_features["turns"],
    }

    # Calculate percentages for each category
    surface_pct = parse_segment_percentages(raw_features["surface"], SURFACE_MAP)
    waytype_pct = parse_segment_percentages(raw_features["waytype"], WAYTYPE_MAP)
    steepness_pct = parse_segment_percentages(raw_features["steepness"], STEEPNESS_MAP)

    # Add all surface percentages (fill missing with 0)
    for surface in SURFACE_MAP.values():
        processed[surface] = surface_pct.get(surface, 0.0)

    # Add all waytype percentages
    for waytype in WAYTYPE_MAP.values():
        processed[waytype] = waytype_pct.get(waytype, 0.0)

    # Rename waytype "Unknown" to "Unknown.1" to match training data
    # (surface already has "Unknown")
    if "Unknown" in [WAYTYPE_MAP[k] for k in WAYTYPE_MAP]:
        processed["Unknown.1"] = processed.pop("Unknown")
        # Re-add surface Unknown
        processed["Unknown"] = surface_pct.get("Unknown", 0.0)

    # Add all steepness percentages
    for steepness in STEEPNESS_MAP.values():
        processed[steepness] = steepness_pct.get(steepness, 0.0)

    return processed


def engineer_features(processed_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply feature engineering transformations.

    Replicates the ORIGINAL logic from 3).Feature Engineering.ipynb.
    This matches what the scaler.pkl and model.pkl were trained on.

    Args:
        processed_features: Dict from process_single_route()

    Returns:
        Dict with engineered features ready for model input
    """
    features = processed_features.copy()

    # Add Turn Density
    features["Turn_Density"] = (
        features["turns"] / (features["distance_m"] / 1000)
        if features["distance_m"] > 0 else 0
    )

    # Surface engineering (OLD approach matching the scaler)
    features["on_road"] = features.get("Asphalt", 0) + features.get("Concrete", 0)
    features["off_road"] = (
        features.get("Dirt", 0) +
        features.get("Grass", 0) +
        features.get("Sand", 0) +
        features.get("Ground", 0) +
        features.get("Unpaved", 0)
    )
    features["Gravel_Tracks"] = (
        features.get("Gravel", 0) + features.get("Compacted Gravel", 0)
    )
    features["Paved_Paths"] = (
        features.get("Paved", 0) + features.get("Paving Stones", 0)
    )
    features["Other"] = (
        features.get("Wood", 0) +
        features.get("Metal", 0) +
        features.get("Grass Paver", 0)
    )
    features["Unknown Surface"] = features.get("Unknown", 0)

    # Waytype engineering (OLD approach matching the scaler)
    features["Paved_Road"] = features.get("Road", 0) + features.get("Street", 0)
    features["Pedestrian"] = features.get("Steps", 0) + features.get("Footway", 0)
    features["Unknown_Way"] = features.get("Unknown.1", 0)
    features["Cycle Track"] = features.get("Path", 0) + features.get("Track", 0)
    features["Other"] += features.get("Ferry", 0) + features.get("Construction", 0)
    features["Main Road"] = features.get("State Road", 0)

    # Steepness engineering (grouped, matching the scaler)
    features["Steep Section"] = (
        features.get("uphill_steep (5% to 7%)", 0) +
        features.get("uphill_very_steep (7% to 10%)", 0) +
        features.get("uphill_extreme (>10%)", 0)
    )
    features["Moderate Section"] = (
        features.get("uphill_gentle (0% to 3%)", 0) +
        features.get("uphill_moderate (3% to 5%)", 0)
    )
    features["Flat Section"] = features.get("flat (0%)", 0)
    features["Downhill Section"] = (
        features.get("downhill_gentle (-5% to 0%)", 0) +
        features.get("downhill_moderate (-7% to -5%)", 0)
    )
    features["Steep Downhill Section"] = (
        features.get("downhill_steep (-10% to -7%)", 0) +
        features.get("downhill_very_steep (-15% to -10%)", 0) +
        features.get("downhill_extreme (<-15%)", 0)
    )

    # Return only the features the scaler expects (in the exact order)
    final_features = {
        "distance_m": features["distance_m"],
        "duration_s": features["duration_s"],
        "ascent_m": features["ascent_m"],
        "descent_m": features["descent_m"],
        "steps": features["steps"],
        "turns": features["turns"],
        "Cycleway": features.get("Cycleway", 0),
        "Turn_Density": features["Turn_Density"],
        "on_road": features["on_road"],
        "off_road": features["off_road"],
        "Gravel_Tracks": features["Gravel_Tracks"],
        "Paved_Paths": features["Paved_Paths"],
        "Other": features["Other"],
        "Unknown Surface": features["Unknown Surface"],
        "Paved_Road": features["Paved_Road"],
        "Pedestrian": features["Pedestrian"],
        "Unknown_Way": features["Unknown_Way"],
        "Cycle Track": features["Cycle Track"],
        "Main Road": features["Main Road"],
        "Steep Section": features["Steep Section"],
        "Moderate Section": features["Moderate Section"],
        "Flat Section": features["Flat Section"],
        "Downhill Section": features["Downhill Section"],
        "Steep Downhill Section": features["Steep Downhill Section"],
    }

    return final_features
