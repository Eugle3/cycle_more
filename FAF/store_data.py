from google.cloud import storage
from google.cloud import bigquery
import os

def upload_json_folder_to_gcs(
    local_folder: str,
    bucket_name: str,
    destination_folder: str,
    area_slug: str = None
):
    """
    Uploads all JSON files from a local folder to a folder in a Google Cloud Storage bucket.

    Args:
        local_folder (str): Path to the local folder containing JSON files.
        bucket_name (str): Name of the Google Cloud Storage bucket.
        destination_folder (str): Folder path inside the bucket (prefix).
                                  Example: "data/json-files"
        area_slug (str, optional): If provided, only upload JSON files whose
            names start with "<area_slug>_". Useful when multiple areas share
            a folder. Defaults to uploading all JSONs.

    Returns:
        None
    """
    # Initialize client
    client = storage.Client(project="cyclemore")
    bucket = client.bucket(bucket_name)

    # Ensure folder path ends correctly
    destination_folder = destination_folder.strip("/")

    # Loop over all files
    for filename in os.listdir(local_folder):
        if not filename.endswith(".json"):
            continue

        if area_slug and not filename.startswith(f"{area_slug}_"):
            continue

        local_path = os.path.join(local_folder, filename)
        blob_path = f"{destination_folder}/{filename}"  # Path in bucket

        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)

        print(f"Uploaded: {local_path} → gs://{bucket_name}/{blob_path}")


def upload_file_to_gcs(local_path: str, bucket_name: str, destination_path: str):
    """
    Upload a single file to a specified path in a GCS bucket.
    """
    client = storage.Client(project="cyclemore")
    bucket = client.bucket(bucket_name)

    destination_path = destination_path.strip("/")

    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)

    print(f"Uploaded: {local_path} → gs://{bucket_name}/{destination_path}")


def upload_csv_to_bigquery(
    csv_path: str,
    dataset_id: str,
    table_id: str,
    project_id: str = None
):
    """
    Uploads a CSV file into a BigQuery table.
    Creates the table automatically if it does not exist.

    Args:
        csv_path (str): Path to the local CSV file.
        dataset_id (str): Name of the BigQuery dataset.
        table_id (str): Name of the BigQuery table.
        project_id (str, optional): GCP project ID.
    """

    client = bigquery.Client(project=project_id)

    table_ref = client.dataset(dataset_id).table(table_id)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,       # ignore header
        autodetect=True,           # detect schema automatically
        write_disposition="WRITE_TRUNCATE"  # replace table on upload
    )

    print(f"📤 Uploading CSV: {csv_path}")
    print(f"➡️  Destination: {dataset_id}.{table_id}\n")

    with open(csv_path, "rb") as file_obj:
        load_job = client.load_table_from_file(
            file_obj,
            table_ref,
            job_config=job_config
        )

    load_job.result()   # Wait until job finishes

    print(f"✅ Upload complete: {dataset_id}.{table_id}")

if __name__ == "__main__":
    # Example invocation for local testing.
    upload_csv_to_bigquery(
        csv_path="amsterdam_processed_routes.csv",
        dataset_id="cycling_routes",
        table_id="amsterdam1",
        project_id="cyclemore"
    )
