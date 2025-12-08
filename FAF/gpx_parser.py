"""
GPX file parser to extract coordinates for route processing.

Handles GPX tracks, routes, and waypoints with smart sampling
to stay within ORS API limits (max 70 waypoints).
"""

import math
import xml.etree.ElementTree as ET
from typing import List, Tuple


def extract_coordinates_from_gpx(gpx_content: bytes, max_waypoints: int = 70) -> List[List[float]]:
    """
    Extract coordinates from GPX file with smart sampling.

    Uses direction-change detection to preserve route character while
    staying under ORS API limit of 70 waypoints.

    Args:
        gpx_content: Bytes content of the GPX file
        max_waypoints: Maximum number of waypoints to return (default 70)

    Returns:
        List of [lon, lat] coordinate pairs (max 70)

    Raises:
        ValueError: If no valid coordinates found in GPX
    """
    # Decode bytes to string
    if isinstance(gpx_content, bytes):
        gpx_content = gpx_content.decode('utf-8')

    # Parse XML
    root = ET.fromstring(gpx_content)

    # GPX namespace (try both with and without namespace)
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    # Try to find track points first (most common for recorded rides)
    track_points = root.findall('.//gpx:trkpt', ns)
    if not track_points:
        # Try without namespace
        track_points = root.findall('.//trkpt')

    if track_points:
        points = _extract_points_list(track_points)
        return _smart_sample(points, max_waypoints)

    # Try route points
    route_points = root.findall('.//gpx:rtept', ns)
    if not route_points:
        route_points = root.findall('.//rtept')

    if route_points:
        points = _extract_points_list(route_points)
        return _smart_sample(points, max_waypoints)

    # Try waypoints
    waypoints = root.findall('.//gpx:wpt', ns)
    if not waypoints:
        waypoints = root.findall('.//wpt')

    if waypoints:
        points = _extract_points_list(waypoints)
        return _smart_sample(points, max_waypoints)

    raise ValueError("No valid track, route, or waypoint data found in GPX file")


def _extract_points_list(points: List[ET.Element]) -> List[List[float]]:
    """
    Extract coordinate pairs from XML point elements.

    Args:
        points: List of XML elements with 'lat' and 'lon' attributes

    Returns:
        List of [lon, lat] pairs (ORS expects lon/lat order)
    """
    coords = []
    for pt in points:
        lat = float(pt.get('lat'))
        lon = float(pt.get('lon'))
        coords.append([lon, lat])  # ORS needs [lon, lat] order
    return coords


def _smart_sample(points: List[List[float]], max_points: int) -> List[List[float]]:
    """
    Smart sampling that preserves route character by keeping significant turns.

    Strategy:
    1. Always keep start and end
    2. Calculate bearing change at each point
    3. Keep points with significant direction changes (turns)
    4. If still over limit, evenly sample remaining points

    Args:
        points: List of [lon, lat] coordinate pairs
        max_points: Maximum number of points to return

    Returns:
        List of sampled [lon, lat] coordinate pairs
    """
    if len(points) <= max_points:
        return points

    # Always keep first and last
    sampled = [0, len(points) - 1]

    # Calculate bearing changes for each point
    bearing_changes = []
    for i in range(1, len(points) - 1):
        prev = points[i - 1]
        curr = points[i]
        next_pt = points[i + 1]

        # Calculate bearing change
        bearing1 = _calculate_bearing(prev, curr)
        bearing2 = _calculate_bearing(curr, next_pt)
        change = abs(_angle_difference(bearing1, bearing2))

        bearing_changes.append((i, change))

    # Sort by bearing change (highest first = sharpest turns)
    bearing_changes.sort(key=lambda x: x[1], reverse=True)

    # Keep points with significant turns
    # We need max_points - 2 more points (already have start/end)
    needed = max_points - 2

    # Take top N points by bearing change
    for idx, _ in bearing_changes[:needed]:
        sampled.append(idx)

    # Sort indices to maintain route order
    sampled.sort()

    return [points[i] for i in sampled]


def _calculate_bearing(point1: List[float], point2: List[float]) -> float:
    """
    Calculate bearing between two points in degrees.

    Args:
        point1: [lon, lat]
        point2: [lon, lat]

    Returns:
        Bearing in degrees (0-360)
    """
    lon1, lat1 = math.radians(point1[0]), math.radians(point1[1])
    lon2, lat2 = math.radians(point2[0]), math.radians(point2[1])

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


def _angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest difference between two angles.

    Args:
        angle1: First angle in degrees
        angle2: Second angle in degrees

    Returns:
        Smallest difference in degrees (-180 to 180)
    """
    diff = angle2 - angle1
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff
