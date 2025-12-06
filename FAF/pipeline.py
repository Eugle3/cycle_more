import os
from FAF.call_turbo import call_turbo, extract_turbo_route
from FAF.store_data import (
    upload_json_folder_to_gcs,
    upload_file_to_gcs,
    upload_csv_to_bigquery,
)
from FAF.call_ors import call_ors, extract_ors_features, load_bigquery_table

def run_turbo_pipeline(
    area_name: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    bucket_name="cycle_more_bucket",
    gcs_turbo_prefix="turbo_data",
    bq_dataset="turbo_coordinates",
    project_id="cyclemore",
):
    safe_area = area_name.replace(" ", "_").lower()
    raw_dir = f"{safe_area}_turbo"
    os.makedirs(raw_dir, exist_ok=True)

    print("\n===== TURBO PIPELINE =====")

    # 1. Call Turbo API
    td = call_turbo(start_lat, start_lon, end_lat, end_lon, area_name)

    # 2. Extract CSV locally
    turbo_csv = f"{safe_area}_turbo.csv"
    df_turbo = extract_turbo_route(td, turbo_csv)

    # 3. Upload Turbo JSONs to GCS
    upload_json_folder_to_gcs(
        local_folder=raw_dir,
        bucket_name=bucket_name,
        destination_folder=f"{gcs_turbo_prefix}/{safe_area}",
    )

    # 4. Upload Turbo CSV → BigQuery
    upload_csv_to_bigquery(
        csv_path=turbo_csv,
        dataset_id=bq_dataset,
        table_id=safe_area,  # table becomes turbo_coordinates.amsterdam
        project_id=project_id,
    )

    print("TURBO PIPELINE COMPLETE.\n")
    return df_turbo


def run_ors_pipeline(
    area_name: str,
    bucket_name="cycle_more_bucket",
    gcs_ors_prefix="raw_ors_data",
    bq_dataset_turbo="turbo_coordinates",
    bq_dataset_ors="cycling_routes",
    project_id="cyclemore",
    start_index=0,     # Start index for route batch
    end_index=None,    # End index for route batch (None = process all)
):

    safe_area = area_name.replace(" ", "_").lower()
    raw_dir = "raw_ors_responses"
    os.makedirs(raw_dir, exist_ok=True)

    print("\n===== ORS PIPELINE =====")

    # -------------------------------------
    # 1. Load Turbo coords from BigQuery
    # -------------------------------------
    print(f"📥 Loading Turbo coords from BigQuery table: {safe_area}")
    from FAF.call_ors import load_bigquery_table  # local import for safety

    coords_df = load_bigquery_table(
        dataset_id=bq_dataset_turbo,
        table_id=safe_area,
        project_id=project_id,
    )

    if coords_df.empty:
        raise ValueError(f"❌ ERROR: Turbo table '{safe_area}' is EMPTY in BigQuery!")

    print(f"📌 Loaded {len(coords_df)} total routes from BigQuery")

    # Slice routes based on start and end index
    if end_index is None:
        end_index = len(coords_df)

    coords_df = coords_df.iloc[start_index:end_index]

    print(f"📌 Processing routes {start_index} to {end_index} ({len(coords_df)} routes)\n")

    # -------------------------------------
    # 2. Call ORS per route
    # -------------------------------------
    print("📡 Calling ORS API for each Turbo route...")
    call_ors(coords_df, area_name=area_name, raw_dir=raw_dir, sleep_seconds=1.0)

    # -------------------------------------
    # 3. Upload ORS JSON to GCS (filtered)
    # -------------------------------------
    print("☁️ Uploading ORS JSON files → GCS…")
    upload_json_folder_to_gcs(
        local_folder=raw_dir,
        bucket_name=bucket_name,
        destination_folder=f"{gcs_ors_prefix}/{safe_area}",
    )

    # -------------------------------------
    # 4. Extract ORS CSV features
    # -------------------------------------
    print("📘 Extracting ORS features → CSV")
    ors_csv = f"{safe_area}_processed_routes.csv"
    ors_df = extract_ors_features(area_name, raw_dir=raw_dir, save_csv=True)

    # Validate CSV
    if ors_df.empty:
        raise ValueError(f"❌ ERROR: ORS results are empty for {area_name}. No data uploaded.")

    print(f"📁 ORS CSV created: {ors_csv} ({len(ors_df)} rows)\n")

    # -------------------------------------
    # 6. Upload CSV → BigQuery
    # -------------------------------------
    print(f"📤 Uploading ORS CSV → BigQuery table: {safe_area}")
    upload_csv_to_bigquery(
        csv_path=ors_csv,
        dataset_id=bq_dataset_ors,
        table_id=safe_area,
        project_id=project_id,
    )

    print("\n✅ ORS PIPELINE COMPLETE\n")


if __name__ == "__main__":
    # Run ORS Pipeline for Netherlands - Process routes 0 to 2000 (Day 1)
    # Tomorrow: change to start_index=2000, end_index=4000
    # Next day: change to start_index=4000, end_index=6000, etc.
    run_ors_pipeline(
        area_name="Tile_ITA_Sicilia",)
