import json
import requests
import pandas as pd
import os
import time
import openrouteservice


def call_turbo(start_lat, start_lon, end_lat, end_lon, area_name):
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

    # ---- Save JSON File ----
    # Clean area_name to avoid illegal filename characters
    safe_area = area_name.replace(" ", "_").lower()
    out_dir = f"{safe_area}_turbo"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Save each route as its own JSON file ----
    elements = data.get("elements", [])
    saved_count = 0

    for element in elements:
        if element.get("type") != "relation":
            continue

        route_id = element.get("id")
        filename = f"{out_dir}/route_{route_id}.json"

        with open(filename, "w") as f:
            json.dump(element, f, indent=2)

        saved_count += 1

    print(f"✅ Saved {saved_count} Turbo routes to folder: {out_dir}")

    return data


def extract_turbo_route(data, csv_path):
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

    routes_df.to_csv(csv_path, index=False)
    print(f"Saved route summary CSV to: {csv_path}")

    return routes_df

def call_ors(df, area_name, raw_dir="raw_ors_responses", sleep_seconds=1.0):
    """
    Calls ORS Directions API for each row in a dataframe and saves raw JSON responses.

    Args:
        df (pd.DataFrame): Contains columns:
            ['id', 'start_lon', 'start_lat', 'end_lon', 'end_lat']
        area_name (str): Name of the area, e.g. "Tokyo"
        raw_dir (str): Directory to save route JSONs.
        sleep_seconds (float): Wait time between API calls.

    Saves files like:
        raw_dir/Tokyo_route_12345.json
    """

    client = openrouteservice.Client(key=os.environ["AK"])

    # Create output directory
    os.makedirs(raw_dir, exist_ok=True)

    # Clean the area name for safe filenames
    safe_area = area_name.replace(" ", "_")

    for i, row in df.iterrows():
        route_id = row["id"]

        start = [row["start_lon"], row["start_lat"]]
        end   = [row["end_lon"], row["end_lat"]]

        try:
            # --- Call ORS API ---
            route = client.directions(
                coordinates=[start, end],
                profile="cycling-regular",
                format="geojson",
                elevation=True,
                instructions=True,
                extra_info=["surface", "waytype", "waycategory", "steepness"],
            )

            # --- Build filename with area name ---
            filename = f"{raw_dir}/{safe_area}_route_{route_id}.json"

            # --- Save JSON ---
            with open(filename, "w") as f:
                json.dump(route, f, indent=2)

            print(f"✅ Saved {safe_area}_route_{route_id}  ({i+1}/{len(df)})")

            if sleep_seconds:
                time.sleep(sleep_seconds)

        except Exception as e:
            print(f"❌ Error on route {route_id}: {e}")
            continue
    return safe_area


def extract_ors_features(area_name, raw_dir="raw_ors_responses", save_csv=True):
    """
    Reads all ORS route JSONs for a given area, extracts summary information,
    returns a dataframe, and optionally saves it as a CSV.

    JSON filenames must follow the format:
        <AreaName>_route_<id>.json

    Args:
        area_name (str): Name of the area used in filenames, e.g. "Tokyo".
        raw_dir (str): Directory containing the JSON files.
        save_csv (bool): If True, saves the output CSV using area_name.

    Returns:
        pd.DataFrame: Extracted route summary dataframe.
    """

    safe_area = area_name.replace(" ", "_")
    results = []

    # Loop through all JSON files for this area
    for filename in os.listdir(raw_dir):

        # Only process files that start with the area name and end in .json
        if not filename.startswith(safe_area) or not filename.endswith(".json"):
            continue

        filepath = os.path.join(raw_dir, filename)

        try:
            with open(filepath, "r") as f:
                route = json.load(f)

            props = route["features"][0]["properties"]
            summary = props["summary"]
            extras = props.get("extras", {})
            segments = props.get("segments", [{}])
            steps = segments[0].get("steps", [])

            # Parse route ID from filename
            # Example: "Tokyo_route_168466.json"
            route_id = filename.replace(f"{safe_area}_route_", "").replace(".json", "")

            # Count turns (ORS uses step['type'] codes 0–7 for turning instructions)
            turn_steps = [s for s in steps if s.get("type") in range(8)]
            num_turns = len(turn_steps)
            num_steps = len(steps)

            results.append({
                "id": route_id,
                "distance_m": summary.get("distance"),
                "duration_s": summary.get("duration"),
                "ascent_m": props.get("ascent"),
                "descent_m": props.get("descent"),
                "steps": num_steps,
                "turns": num_turns,
                "surface": extras.get("surface", {}).get("values", []),
                "waytype": extras.get("waytype", {}).get("values", []),
                "waycategory": extras.get("waycategory", {}).get("values", []),
                "steepness": extras.get("steepness", {}).get("values", []),
            })

            print(f"✅ Parsed {filename}")

        except Exception as e:
            print(f"❌ Error parsing {filename}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save CSV if requested
    if save_csv:
        csv_name = f"{safe_area}_processed_routes.csv"
        df.to_csv(csv_name, index=False)
        print(f"\n📁 Saved processed data to: {csv_name}")

    return df


if __name__ == "__main__":
    # Example main guard usage: uncomment the call you want during local runs.
    # Running this module directly will perform a network request.
    api_call = call_turbo(52.3660, 4.8850, 52.3740, 4.9000, "Amsterdam")

    # Extract and print the summaries so a developer running the script can inspect results quickly.
    summaries = extract_turbo_route(api_call, "amsterdam_test.csv")
    ors = call_ors(summaries, "amsterdam", raw_dir="amsterdam_raw_ors_responses", sleep_seconds=1.0)
    ors_features = extract_ors_features("amsterdam", raw_dir="amsterdam_raw_ors_responses", save_csv=True)

    print("Route summaries:")
    print(summaries)
    print("ORS Features:")
    print(ors_features)
