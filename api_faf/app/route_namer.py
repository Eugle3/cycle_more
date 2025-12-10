"""
Auto-naming for routes with generic names like "unnamed route".

Uses reverse geocoding to generate names like:
- "Richmond to Kingston" (point-to-point)
- "Loop from Richmond" (circular routes)
"""

import time
from typing import Optional, Tuple
import requests
from google.cloud import storage


def fetch_route_coordinates(route_id: int, bucket: str = "cycle_more_bucket", prefix: str = "all_routes/") -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Fetch start and end coordinates for a route from GCS.

    Returns:
        ((start_lat, start_lon), (end_lat, end_lon)) or None if not found
    """
    try:
        client = storage.Client()
        bucket_obj = client.bucket(bucket)
        blob_path = f"{prefix}{route_id}.json"
        blob = bucket_obj.blob(blob_path)

        if not blob.exists():
            return None

        import json
        data = json.loads(blob.download_as_text())

        if not data or "coordinates" not in data or not data["coordinates"]:
            return None

        coords = data["coordinates"]
        start = (coords[0][1], coords[0][0])  # (lat, lon)
        end = (coords[-1][1], coords[-1][0])

        return (start, end)

    except Exception as e:
        print(f"Error fetching coordinates for route {route_id}: {e}")
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Use Nominatim API to get place name from coordinates.

    Returns place name like "Richmond" or "Kingston upon Thames" or None.
    """
    try:
        # Nominatim requires a user agent
        headers = {
            "User-Agent": "CycleMore/1.0 (route naming service)"
        }

        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 14,  # City/town level
        }

        response = requests.get(url, params=params, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()

            # Try to extract a good place name
            address = data.get("address", {})

            # Prefer: suburb > village > town > city
            place_name = (
                address.get("suburb") or
                address.get("village") or
                address.get("town") or
                address.get("city") or
                address.get("county") or
                None
            )

            return place_name

        return None

    except Exception as e:
        print(f"Geocoding error for ({lat}, {lon}): {e}")
        return None


def is_loop(start: Tuple[float, float], end: Tuple[float, float], threshold_km: float = 0.5) -> bool:
    """
    Check if start and end points are close enough to be considered a loop.

    Uses Haversine formula to calculate distance.
    """
    from math import radians, sin, cos, sqrt, atan2

    lat1, lon1 = start
    lat2, lon2 = end

    # Haversine formula
    R = 6371  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance < threshold_km


def generate_route_name(route_id: int, bucket: str = "cycle_more_bucket") -> Optional[str]:
    """
    Generate a human-readable route name based on start/end locations.

    Returns:
        - "Loop from Richmond" (if circular)
        - "Richmond to Kingston" (if point-to-point)
        - None (if geocoding fails)
    """
    # Fetch coordinates
    coords = fetch_route_coordinates(route_id, bucket=bucket)
    if not coords:
        print(f"Route {route_id}: Failed to fetch coordinates from GCS")
        return None

    start, end = coords
    print(f"Route {route_id}: Start coords {start}, End coords {end}")

    # Check if it's a loop
    is_circular = is_loop(start, end)
    print(f"Route {route_id}: Is circular? {is_circular}")

    # Geocode start point
    start_place = reverse_geocode(start[0], start[1])
    print(f"Route {route_id}: Start place = {start_place}")

    # Respect Nominatim rate limit (1 req/sec)
    time.sleep(1.1)

    if is_circular:
        if start_place:
            return f"Loop from {start_place}"
        else:
            print(f"Route {route_id}: Circular route but start_place is None")
            return None

    # Point-to-point: geocode end point too
    end_place = reverse_geocode(end[0], end[1])
    print(f"Route {route_id}: End place = {end_place}")

    if start_place and end_place:
        if start_place == end_place:
            return f"Loop from {start_place}"
        else:
            return f"{start_place} to {end_place}"
    elif start_place:
        return f"Route from {start_place}"
    elif end_place:
        return f"Route to {end_place}"
    else:
        print(f"Route {route_id}: Both start_place and end_place are None")
        return None


def enhance_route_name(
    route_id: int,
    current_name: str,
    bucket: str = "cycle_more_bucket",
    distance_m: float = None,
    ascent_m: float = None
) -> str:
    """
    Enhance a route name if it's generic (like "unnamed route").

    Args:
        route_id: Route ID
        current_name: Current route name from database
        bucket: GCS bucket name
        distance_m: Route distance in meters (for fallback naming)
        ascent_m: Route ascent in meters (for fallback naming)

    Returns:
        Enhanced name if current name is generic, otherwise returns current name
    """
    # List of generic names that should be enhanced
    generic_patterns = [
        "unnamed route",
        "unknown route",
        "untitled",
        "no name",
        "unnamed",
    ]

    # Check if current name is generic (case-insensitive)
    name_lower = current_name.lower().strip()

    # Check exact matches or if the entire name is just "route"
    is_generic = (
        name_lower in generic_patterns or
        name_lower == "route" or
        any(pattern == name_lower for pattern in generic_patterns)
    )

    if not is_generic:
        print(f"Route {route_id} name '{current_name}' is not generic, skipping enhancement")
        return current_name

    print(f"Route {route_id} has generic name '{current_name}', attempting to enhance...")

    # Try to generate a better name using geocoding
    try:
        new_name = generate_route_name(route_id, bucket=bucket)

        if new_name:
            print(f"Route {route_id} enhanced: '{current_name}' -> '{new_name}'")
            return new_name
    except Exception as e:
        print(f"Error enhancing route {route_id}: {e}")

    # Fallback: Use distance/ascent to create a descriptive name
    if distance_m is not None and ascent_m is not None:
        distance_km = distance_m / 1000
        if ascent_m > 500:
            terrain = "hilly"
        elif ascent_m > 200:
            terrain = "rolling"
        else:
            terrain = "flat"

        fallback_name = f"{distance_km:.0f}km {terrain} route"
        print(f"Route {route_id} using fallback name: '{fallback_name}'")
        return fallback_name

    print(f"Route {route_id} geocoding failed and no fallback data, keeping original name")
    return current_name
