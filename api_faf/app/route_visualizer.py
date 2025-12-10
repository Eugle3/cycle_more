"""Route visualization using Folium maps, based on route_recommendation_viz notebook."""

import json
from typing import List, Tuple, Dict, Any


def extract_coords_from_route_json(route_json: Dict[str, Any]) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Extract (lat, lon) coordinates and elevation data from route JSON.

    Handles both FeatureCollection format and direct geometry format.

    Args:
        route_json: Route JSON dictionary

    Returns:
        Tuple of (coords_list, elevations_list)
        - coords_list: List of (lat, lon) tuples
        - elevations_list: List of elevation values in meters
    """
    coords = []
    elevations = []

    # Handle FeatureCollection format
    if route_json.get('type') == 'FeatureCollection':
        features = route_json.get('features', [])
        for feature in features:
            geometry = feature.get('geometry', {})
            geom_type = geometry.get('type')
            coordinates = geometry.get('coordinates', [])

            if geom_type == 'LineString':
                for coord in coordinates:
                    lon, lat = coord[0], coord[1]
                    coords.append((lat, lon))
                    if len(coord) > 2:
                        elevations.append(coord[2])

            elif geom_type == 'MultiLineString':
                for line in coordinates:
                    for coord in line:
                        lon, lat = coord[0], coord[1]
                        coords.append((lat, lon))
                        if len(coord) > 2:
                            elevations.append(coord[2])

    # Handle direct geometry format
    elif 'geometry' in route_json:
        geometry = route_json['geometry']
        geom_type = geometry.get('type')
        coordinates = geometry.get('coordinates', [])

        if geom_type == 'LineString':
            for coord in coordinates:
                lon, lat = coord[0], coord[1]
                coords.append((lat, lon))
                if len(coord) > 2:
                    elevations.append(coord[2])

        elif geom_type == 'MultiLineString':
            for line in coordinates:
                for coord in line:
                    lon, lat = coord[0], coord[1]
                    coords.append((lat, lon))
                    if len(coord) > 2:
                        elevations.append(coord[2])

    return coords, elevations


def load_route_json_from_gcs(route_id: int, bucket: str = "cycle_more_bucket", prefix: str = "all_routes/") -> Dict[str, Any]:
    """
    Load route JSON from GCS by route ID.

    Args:
        route_id: Route ID to load
        bucket: GCS bucket name
        prefix: GCS folder prefix

    Returns:
        Route JSON dictionary
    """
    # Lazy imports
    from google.cloud import storage
    import fsspec

    # Find route file
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    search_pattern = f"route_{route_id}.json"

    matches = [
        blob.name
        for blob in client.list_blobs(bucket_obj, prefix=prefix)
        if blob.name.endswith(search_pattern)
    ]

    if not matches:
        raise FileNotFoundError(f"Route {route_id} not found in GCS")

    path = matches[0]

    # Load JSON using fsspec
    gcs_path = f"gs://{bucket}/{path}"
    with fsspec.open(gcs_path, 'r', token="google_default") as f:
        route_json = json.load(f)

    return route_json


def create_folium_map(
    coords: List[Tuple[float, float]],
    route_name: str = "Route",
    route_id: int = None,
    distance_km: float = None,
    ascent_m: float = None
) -> str:
    """
    Create an interactive Folium map with a single route.

    Args:
        coords: List of (lat, lon) tuples
        route_name: Name of the route for display
        route_id: Route ID (optional)
        distance_km: Route distance in km (optional)
        ascent_m: Route ascent in meters (optional)

    Returns:
        HTML string of the Folium map
    """
    # Lazy import
    import folium

    if not coords:
        raise ValueError("No coordinates provided")

    # Calculate center point
    center_lat = sum(coord[0] for coord in coords) / len(coords)
    center_lon = sum(coord[1] for coord in coords) / len(coords)

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

    # Build popup text with optional info
    popup_parts = [f"<b>{route_name}</b>"]
    if route_id:
        popup_parts.append(f"ID: {route_id}")
    if distance_km:
        popup_parts.append(f"Distance: {distance_km:.1f} km")
    if ascent_m:
        popup_parts.append(f"Ascent: {ascent_m:.0f} m")

    popup_text = "<br>".join(popup_parts)

    # Add route polyline
    folium.PolyLine(
        coords,
        color='#FF6B6B',
        weight=5,
        opacity=0.85,
        popup=popup_text
    ).add_to(m)

    # Add start marker (green)
    folium.Marker(
        coords[0],
        popup='<b>Start</b>',
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)

    # Add end marker (red)
    folium.Marker(
        coords[-1],
        popup='<b>End</b>',
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)

    # Return HTML
    return m._repr_html_()


def visualize_route(
    route_id: int,
    route_name: str = "Route",
    distance_km: float = None,
    ascent_m: float = None,
    bucket: str = "cycle_more_bucket"
) -> str:
    """
    Main function to visualize a single route.

    Loads route from GCS, extracts coordinates, and creates Folium map.

    Args:
        route_id: Route ID to visualize
        route_name: Name of the route
        distance_km: Distance in kilometers (optional)
        ascent_m: Ascent in meters (optional)
        bucket: GCS bucket name

    Returns:
        HTML string of the Folium map

    Raises:
        FileNotFoundError: If route not found in GCS
        ValueError: If route has no coordinates
    """
    # Load route JSON from GCS
    route_json = load_route_json_from_gcs(route_id, bucket)

    # Extract coordinates
    coords, elevations = extract_coords_from_route_json(route_json)

    if not coords:
        raise ValueError(f"No coordinates found for route {route_id}")

    # Create and return Folium map HTML
    return create_folium_map(
        coords=coords,
        route_name=route_name,
        route_id=route_id,
        distance_km=distance_km,
        ascent_m=ascent_m
    )
