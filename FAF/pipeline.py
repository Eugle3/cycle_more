import os
from FAF.call_turbo import call_turbo, extract_turbo_route
from FAF.store_data import upload_json_folder_to_gcs, upload_csv_to_bigquery


def run_turbo_pipeline(
    area_name: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    bucket_name: str = "cycle_more_bucket",
    gcs_turbo_prefix: str = "turbo_data",
    bq_project: str = "cyclemore",
    bq_dataset_turbo: str = "turbo_coordinates",
):
    """
    Full end-to-end pipeline for Turbo processing.
    """

    safe_area_slug = area_name.replace(" ", "_").lower()

    print("\n============================")
    print(f"🚀 Starting pipeline for: {area_name}")
    print("============================\n")

    # -----------------------------
    # 1. Call Turbo API
    # -----------------------------
    print("📡 Calling Turbo API...")
    turbo_data = call_turbo(start_lat, start_lon, end_lat, end_lon, area_name)

    # -----------------------------
    # 2. Extract Turbo dataframe
    # -----------------------------
    turbo_csv = f"{safe_area_slug}_turbo.csv"
    print("📄 Extracting Turbo route summary...")
    turbo_df = extract_turbo_route(turbo_data, turbo_csv)

    # -----------------------------
    # 3. Upload Turbo JSON folder → GCS
    # -----------------------------
    turbo_json_folder = f"{safe_area_slug}_turbo"
    gcs_turbo_folder = f"{gcs_turbo_prefix}/{safe_area_slug}"

    print("☁️ Uploading Turbo JSON to GCS...")
    upload_json_folder_to_gcs(
        local_folder=turbo_json_folder,
        bucket_name=bucket_name,
        destination_folder=gcs_turbo_folder
    )

    # -----------------------------
    # 4. Upload Turbo CSV → BigQuery
    # -----------------------------
    print("📊 Uploading Turbo dataframe to BigQuery...")
    upload_csv_to_bigquery(
        csv_path=turbo_csv,
        dataset_id=bq_dataset_turbo,
        table_id=safe_area_slug,
        project_id=bq_project
    )


if __name__ == "__main__":

    area_dict = {
        "area_name": "Amsterdam",
        "start_lat": 52.3660,
        "start_lon": 4.8850,
        "end_lat": 52.3740,
        "end_lon": 4.9000,
    }

    #build list of area dictionaries containing area name and co ords
    # areas = [
    #     {
    #         "area_name": "Amsterdam",
    #         "start_lat": 52.3660,
    #         "start_lon": 4.8850,
    #         "end_lat": 52.3740,
    #         "end_lon": 4.9000,
    #     },
    #             {
    #         "area_name": "Amsterdam",
    #         "start_lat": 52.3660,
    #         "start_lon": 4.8850,
    #         "end_lat": 52.3740,
    #         "end_lon": 4.9000,
    #     },
    # ]

    for area_dict in areas:
        run_turbo_pipeline(**area_dict)






"""
# -----------------------------
# 5. Call ORS API for each route
# -----------------------------
print("\n🛣 Calling ORS API for each Turbo route...")
call_ors(turbo_df, area_name, raw_dir=raw_ors_dir)

# -----------------------------
# 6. Extract ORS dataframe
# -----------------------------
print("📄 Extracting ORS features...")
ors_df = extract_ors_features(
    area_name=area_name,
    raw_dir=raw_ors_dir,
    save_csv=True
)

ors_csv = f"{safe_area_slug}_processed_routes.csv"

# -----------------------------
# 7. Upload ORS JSON folder → GCS
# -----------------------------
gcs_ors_folder = f"{gcs_ors_prefix}/{safe_area_slug}"

print("☁️ Uploading ORS JSON to GCS...")
upload_json_folder_to_gcs(
    local_folder=raw_ors_dir,
    bucket_name=bucket_name,
    destination_folder=gcs_ors_folder
)

# -----------------------------
# 8. Upload ORS dataframe → BigQuery
# -----------------------------
print("📊 Uploading ORS processed dataframe to BigQuery...")
upload_csv_to_bigquery(
    csv_path=ors_csv,
    dataset_id=bq_dataset_ors,
    table_id=safe_area_slug,
    project_id=bq_project
)

print("\n🎉 DONE! Pipeline completed successfully.")
return turbo_df, ors_df"""
