# ...existing code...
"""
call_turbo.py

Module purpose:
This module queries the Overpass API endpoint hosted at maps.mail.ru to retrieve
OpenStreetMap "relation" objects tagged as bicycle routes within a bounding box.
It provides:
- call_turbo(start_lat, start_lon, end_lat, end_lon): perform the Overpass query
and return parsed JSON.
- extract_turbo_route(data): extract a concise route summary (id, name,
start/end coords, number of points).
- test(): a tiny helper to confirm the module is importable and callable.

Notes on behavior and output:
- call_turbo builds an Overpass QL bounding-box query and posts it to the API.
  It returns the parsed JSON (a Python dict matching Overpass JSON structure)
  and prints the raw response data.
  Possible exceptions: requests.RequestException on network errors,
  ValueError/JSONDecodeError if response isn't valid JSON.
- extract_turbo_route expects the Overpass JSON structure (a dict with key
'elements' containing OSM elements).
  It iterates relation-type elements, collects member geometry segments
  (members with 'geometry'),
  flattens them into a single coordinate list, and generates a list of summaries:
    [{'id': int, 'name': str, 'start_lon': float, 'start_lat': float,
    'end_lon': float, 'end_lat': float, 'num_points': int}, ...]
  If no geometry is found for a relation it is skipped.

Example usage:
    data = call_turbo(52.5200, 13.4050, 53.5206, 15.4094)
    summaries = extract_turbo_route(data)
    # summaries is a list of route summary dicts

Keep network calls and printing in your main guard to avoid side effects on
import.
"""
# ...existing code...
import requests
import pandas as pd

def test():
    """
    Simple smoke test.

    Prints a short message to verify that the module loads and the function is
    callable.
    No inputs, no return value.
    """
    # Informal helper for local development/debugging
    print("This is a test function.")


def call_turbo(start_lat, start_lon, end_lat, end_lon):
    """
    Query the Overpass API for bicycle route relations inside a bounding box.

    Args:
        start_lat (float): minimum latitude of bounding box (south).
        start_lon (float): minimum longitude of bounding box (west).
        end_lat (float): maximum latitude of bounding box (north).
        end_lon (float): maximum longitude of bounding box (east).

    Returns:
        dict: parsed JSON response from the Overpass API (structure follows
        Overpass JSON format).

    Side effects:
        - Sends a POST request to the maps.mail.ru Overpass endpoint.
        - Prints the API response data to stdout for debugging.

    Notes for collaborators:
        - The bounding box order used here is (south, west, north, east) which
        matches Overpass QL.
        - The timeout in the query is set to 900 seconds to allow large bbox
        queries; adjust if necessary.
        - Network errors or invalid JSON will raise exceptions
        (requests.RequestException, ValueError).
    """
    # Overpass interpreter endpoint used by maps.mail.ru
    url = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"

    # Build the Overpass QL query. The triple-quoted string is intentionally formatted with newlines
    # so it is readable in logs and easier to edit. The query filters relations tagged with route=bicycle.
    query = f"""
        [out:json][timeout:900];
        (
        relation["route"="bicycle"]({start_lat},{start_lon},{end_lat},{end_lon});
        );
        out geom;
        """

    # Perform the POST request. We use form-encoded "data" as the Overpass endpoint expects the query in the "data" field.
    response = requests.post(url, data={"data": query})

    # Convert the response body to Python dict (JSON). This may raise if the body isn't valid JSON.
    data = response.json()

    # Print the response for debugging — collaborators can see raw API data in stdout.
    print("Turbo API response data:")
    print(data)

    return data


def extract_turbo_route(data):
    """
    Produce a concise summary list for bicycle route relations from Overpass JSON.

    Args:
        data (dict): Overpass JSON response. Expected to contain 'elements'
        list with OSM elements.

    Returns:
        list of dict: Each dict summarizes a found relation with keys:
            - id: relation id (int)
            - name: relation name tag or 'Unnamed route'
            - start_lon, start_lat: coordinates of the first point
            - end_lon, end_lat: coordinates of the last point
            - num_points: total number of points aggregated from member geometry
            segments

    Behavior details:
        - Only elements with 'type' == 'relation' are considered.
        - For each relation, the function checks members for a 'geometry' key
        (these are arrays of {lat, lon}).
        - It concatenates segment geometries in the order members appear;
        this is a simple flattening,
          and may produce discontinuities for complex relations. This function
          is intended for a quick summary,
          not for reconstructing a fully topologically consistent path.
    """
    route_summary = []

    # 'elements' is the top-level list of OSM elements returned by Overpass
    for route in data.get('elements', []):
        # Skip any non-relation elements (nodes/ways may also be present)
        if route.get('type') != 'relation':
            continue  # skip non-route elements

        # Collect coordinates from all members that include 'geometry'
        coords = []
        for member in route.get('members', []):
            # A member with 'geometry' contains an ordered list of {lat, lon} points
            if 'geometry' in member:
                # Convert each point dict to a (lon, lat) tuple to match common geo ordering (lon, lat)
                segment_coords = [(pt['lon'], pt['lat']) for pt in member['geometry']]
                coords.extend(segment_coords)

        # If we collected any coordinates, produce a summary entry
        if coords:
            start_coord = coords[0]
            end_coord = coords[-1]
            route_summary.append({
                'id': route.get('id'),
                'name': route.get('tags', {}).get('name', 'Unnamed route'),
                'start_lon': start_coord[0],
                'start_lat': start_coord[1],
                'end_lon': end_coord[0],
                'end_lat': end_coord[1],
                'num_points': len(coords)
            })
    # Convert results to DataFrame
    routes_df = pd.DataFrame(route_summary)
    return routes_df

# call_turbo(52.5200, 13.4050, 53.5206, 15.4094)


if __name__ == "__main__":
    # Example main guard usage: uncomment the call you want during local runs.
    # Running this module directly will perform a network request.
    api_call = call_turbo(52.5200, 13.4050, 53.5206, 15.4094)

    # Extract and print the summaries so a developer running the script can inspect results quickly.
    summaries = extract_turbo_route(api_call)
    print("Route summaries:")
    print(summaries)
