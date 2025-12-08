"""
Complete pipeline for processing GPX uploads into model-ready features.

This orchestrates the entire flow:
GPX → Coordinates → ORS API → Raw Features → Processed → Engineered Features
"""

from typing import Dict, Any
import os
import openrouteservice
from dotenv import load_dotenv

from gpx_parser import extract_coordinates_from_gpx
from call_ors import extract_single_route_features
from data_processing import process_single_route, engineer_features

load_dotenv()


def process_gpx_file(gpx_content: bytes, max_waypoints: int = 70) -> Dict[str, Any]:
    """
    Complete pipeline: GPX file → engineered features ready for KNN model.

    Args:
        gpx_content: Bytes content of the GPX file
        max_waypoints: Maximum waypoints for ORS API (default 70)

    Returns:
        Dict of engineered features ready for model input

    Raises:
        ValueError: If GPX parsing fails
        Exception: If ORS API call or processing fails
    """
    # Step 1: Extract coordinates from GPX (with smart sampling)
    coordinates = extract_coordinates_from_gpx(gpx_content, max_waypoints)

    # Step 2: Call ORS API
    ors_response = _call_ors_api(coordinates)

    # Step 3: Extract raw features from ORS response
    raw_features = extract_single_route_features(ors_response)

    # Step 4: Process to percentages
    processed_features = process_single_route(raw_features)

    # Step 5: Apply feature engineering
    engineered_features = engineer_features(processed_features)

    return engineered_features


def _call_ors_api(coordinates: list) -> Dict[str, Any]:
    """
    Call ORS Directions API with coordinates.

    Args:
        coordinates: List of [lon, lat] pairs (max 70)

    Returns:
        ORS API GeoJSON response

    Raises:
        Exception: If ORS API call fails
    """
    client = openrouteservice.Client(key=os.environ["AK"])

    try:
        route = client.directions(
            coordinates=coordinates,
            profile="cycling-regular",
            format="geojson",
            elevation=True,
            instructions=True,
            extra_info=["surface", "waytype", "waycategory", "steepness"],
        )
        return route

    except Exception as e:
        raise Exception(f"ORS API call failed: {str(e)}")
