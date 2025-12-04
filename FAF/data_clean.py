from google.cloud import bigquery
import pandas as pd
import os


def concat_tables(tables :list , Big_table_name : str):
    """ returns df """

    os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    # Initialize client (will use credentials from your environment)
    client = bigquery.Client(project='cyclemore')

    # Verify it's working
    print(f"✅ Connected to project: {client.project}")

    # List tables to see what you have
    all_tables = list(client.list_tables("cycling_routes"))
    print("\n📊 Available tables:")
    for table in all_tables:
        print(f"  - {table.table_id}")

    client = bigquery.Client(project='cyclemore')

    # List all your UK tables
    '''tables = ['UK1', 'UK2_Data', 'UK3', 'UK4', 'UK_5']'''  # Adjust based on what tables you saw above

    if tables is None:
        tables = [table.table_id for table in all_tables]

    dataframes = []
    print("concatenating tables:", tables)
    for table_name in tables:
        query = f"SELECT * FROM `cyclemore.cycling_routes.{table_name}`" #problem with cyclemore.cycling_routes

        try:
            df = client.query(query).to_dataframe()
            #add Table source column
            df["source_table"] = table_name

            dataframes.append(df)
            print(f"✅ Fetched {table_name}: {len(df)} routes, {len(df.columns)} columns")
        except Exception as e:
            print(f"❌ Error fetching {table_name}: {e}")

    # Combine all dataframes
    if dataframes:
        all_routes = pd.concat(dataframes, ignore_index=True) #axis 0 (vertical) by default

        # Remove duplicates by route ID
        print(f"\nBefore deduplication: {len(all_routes)} routes")
        all_routes = all_routes.drop_duplicates(subset=['id'], keep='first')
        print(f"After deduplication: {len(all_routes)} routes")

        # Save combined file
        all_routes.to_csv(f'{Big_table_name}.csv', index=False)
        print(f"\n💾 Saved to {Big_table_name}.csv")

        # View
        print(all_routes.head())
        print(f"\nFinal shape: {all_routes.shape}")

        return all_routes




if __name__ == "__main__":
    concat_df = concat_tables(['UK4', 'UK_5'], 'big_table_test')

    #upload big df to big query

    #concat_df = concat_tables(None, 'all_table_test')


# what still needs to be done :
#       - checking the right. number of columns (13) for each dataframe
#       - uploading to big query (upload_to_bigquery() function)⬇️


#-----------------------------------------------!!!!!!! READ THE WARNING !!!!!!------------------------------------------------------

def upload_to_bigquery(df, table_name, dataset_id='cycling_routes', allow_truncate: bool = False):
    """
    WARNING: overites table by default 
    This function uses WRITE_TRUNCATE by default which WILL REPLACE
    the target table in BigQuery (it deletes existing rows in the table and
    writes the provided DataFrame). Do NOT call this function unless you
    intend to replace the table contents.

    To proceed, call with allow_truncate=True. Without that explicit flag the
    function will abort to prevent accidental data loss.

    Upload a pandas DataFrame directly to Google BigQuery.

    Args:
        df (pandas.DataFrame): DataFrame to upload (should be deduplicated/cleaned).
        table_name (str): Destination BigQuery table name (no project/dataset).
        dataset_id (str): BigQuery dataset name (default: 'cycling_routes').
        allow_truncate (bool): Must be True to permit WRITE_TRUNCATE behavior.

    Notes:
        - WRITE_TRUNCATE will delete existing data in the target table.
        - This function does not perform deduplication; prepare df before calling.
    """
    if not allow_truncate:
        raise RuntimeError(
            "ABORTING: WRITE_TRUNCATE is configured for this upload. "
            "Pass allow_truncate=True to confirm you understand this will "
            "replace the existing table contents."
        )

    # Uses credentials from GOOGLE_APPLICATION_CREDENTIALS environment variable
    client = bigquery.Client(project='cyclemore')

    # Full table identifier in format: project.dataset.table
    table_id = f"cyclemore.{dataset_id}.{table_name}"

    # Configure load job: truncate existing table and autodetect schema
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )

    # Status message
    print(f"⬆️  Uploading {len(df)} rows to {table_id} (WRITE_TRUNCATE enabled)...")

    # Start the upload job and wait for completion
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # wait

    # Confirm success and print table metadata
    print(f"✅ Successfully uploaded to {table_id}")
    table = client.get_table(table_id)
    print(f"📊 Table now has {table.num_rows} rows and {len(table.schema)} columns")

    return table
# ...existing code...
