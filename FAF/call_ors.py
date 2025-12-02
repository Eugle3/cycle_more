import json
import os
import time
from pathlib import Path

import openrouteservice
import pandas as pd


def call_ORS(data, save_path="ors_route_results.json", raw_dir="raw_ors_responses_UK2", sleep_seconds=1.5):
    """
    Call the OpenRouteService Directions API for each route row.

    Args:
        data (DataFrame): rows with id, name, start_lon, start_lat, end_lon, end_lat.
        save_path (str): where to persist the aggregated results.
        raw_dir (str): directory for saving raw ORS responses (debugging).
        sleep_seconds (float): delay between requests to avoid rate limiting.
    """
    client = openrouteservice.Client(key=os.environ["AK"])
    results = []
    save_interval = 10
    call_df = data
    os.makedirs(raw_dir, exist_ok=True)

    for i, row in call_df.iterrows():
        start = (row["start_lon"], row["start_lat"])
        end = (row["end_lon"], row["end_lat"])

        try:
            route = client.directions(
                coordinates=[start, end],
                profile="cycling-regular",
                format="geojson",
                elevation=True,
                instructions=True,
                extra_info=["surface", "waytype", "waycategory", "steepness"],
            )

            # Save raw response per route for debugging/inspection
            raw_filename = f"{raw_dir}/route_{row['id']}.json"
            with open(raw_filename, "w") as raw_file:
                json.dump(route, raw_file, indent=2)

            props = route["features"][0]["properties"]
            summary = props["summary"]
            extras = props.get("extras", {})
            segments = props["segments"]
            steps = segments[0]["steps"]
            turn_steps = [s for s in steps if s["type"] in {0, 1, 2, 3, 4, 5, 6, 7}]
            turns = len(turn_steps)
            steps = len(steps)

            results.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "distance_m": summary.get("distance"),
                    "duration_s": summary.get("duration"),
                    "ascent_m": [props["ascent"]],
                    "descent_m": [props["descent"]],
                    "steps": steps,
                    "turns": turns,
                    "surface": extras.get("surface", {}).get("values", []),
                    "waytype": extras.get("waytype", {}).get("values", []),
                    "waycategory": extras.get("waycategory", {}).get("values", []),
                    "steepness": extras.get("steepness", {}).get("values", []),
                }
            )

            print(f"✅ Route {i + 1} processed successfully")
            if sleep_seconds:
                time.sleep(sleep_seconds)  # rate-limit protection

            # Save partial file
            if (i + 1) % save_interval == 0:
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Saved partial results after {i + 1} routes")

        except Exception as e:
            print(f"❌ Error on route {row['id']}: {e}")

    # always write whatever we collected
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def load_route_data(csv_path=None):
    """
    Load route data CSV. Defaults to the file next to this script.
    """
    if csv_path is None:
        csv_path = Path(__file__).with_name("UK_cycle_routes_summary.csv")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    route_data = load_route_data()
    results = call_ORS(route_data, sleep_seconds=0)  # set to 0 for quick local smoke test
    print(f"Finished. {len(results)} route(s) processed.")
