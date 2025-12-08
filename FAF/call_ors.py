import json
import os
import time
import openrouteservice
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_bigquery_table(
    dataset_id: str,
    table_id: str,
    project_id: str = "cyclemore"
) -> pd.DataFrame:
    """
    Loads a BigQuery table into a pandas DataFrame.

    Args:
        dataset_id (str): BigQuery dataset name.
        table_id (str): BigQuery table name.
        project_id (str): GCP project ID.

    Returns:
        pd.DataFrame: Table contents as a DataFrame.
    """

    client = bigquery.Client(project=project_id)

    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{full_table_id}`"

    print(f"📥 Loading BigQuery table: {full_table_id}")

    df = client.query(query).to_dataframe()

    print(f"✅ Loaded {len(df)} rows from {full_table_id}")

    return df



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

    # Clean the area name for safe, consistent filenames (lowercase)
    safe_area = area_name.replace(" ", "_").lower()

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


def extract_single_route_features(ors_response: dict, route_id: str = "uploaded") -> dict:
    """
    Extract features from a single ORS API response.

    Args:
        ors_response (dict): ORS API GeoJSON response
        route_id (str): Optional route identifier

    Returns:
        dict: Extracted route features with raw ORS data
    """
    props = ors_response["features"][0]["properties"]
    summary = props["summary"]
    extras = props.get("extras", {})
    segments = props.get("segments", [{}])
    steps = segments[0].get("steps", [])

    # Count turns (ORS uses step['type'] codes 0–7 for turning instructions)
    turn_steps = [s for s in steps if s.get("type") in range(8)]
    num_turns = len(turn_steps)
    num_steps = len(steps)

    return {
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
    }


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

    safe_area = area_name.replace(" ", "_").lower()
    results = []

    # Loop through all JSON files for this area
    for filename in os.listdir(raw_dir):

        # Only process files that start with the area name and end in .json
        fname_lower = filename.lower()
        if not fname_lower.startswith(safe_area) or not fname_lower.endswith(".json"):
            continue

        filepath = os.path.join(raw_dir, filename)

        try:
            with open(filepath, "r") as f:
                route = json.load(f)

            # Parse route ID from filename (case-insensitive)
            # Example: "Tokyo_route_168466.json"
            route_id = os.path.splitext(filename)[0].split("_route_")[-1]

            # Use the refactored single-route function
            route_features = extract_single_route_features(route, route_id)
            results.append(route_features)

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
    load_bigquery_table("turbo_coordinates","amsterdam_test","cyclemore")

    # route_data = load_route_data()
    # results = call_ORS(route_data, sleep_seconds=0)  # set to 0 for quick local smoke test
    # print(f"Finished. {len(results)} route(s) processed.")
