# def fetch_routes(tables: list[str], project: str, dataset: str, out_path: str):
#       client = bigquery.Client(project=project)
#       frames = []
#       for name in tables:
#           df = client.query(f"SELECT * FROM `{project}.{dataset}.{name}`").to_dataframe()
#           frames.append(df)
#       all_routes = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
#       all_routes.to_csv(out_path, index=False)

#   def _parse_segment_percentages(data, mapping: dict[int, str]):
#       if isinstance(data, str):
#           data = ast.literal_eval(data)
#       totals = {}
#       total_len = 0
#       for start, end, code in data:
#           seg_len = end - start
#           total_len += seg_len
#           totals[code] = totals.get(code, 0) + seg_len
#       return {mapping.get(code, code): round(length / total_len * 100, 2)
#               for code, length in totals.items()} if total_len else {}

#   def clean_routes(in_path: str, out_path: str):
#       df = pd.read_csv(in_path)
#       df["ascent_m"] = df["ascent_m"].astype(str).str.strip("[]").astype(float)
#       df["descent_m"] = df["descent_m"].astype(str).str.strip("[]").astype(float)
#       df["name"] = df["name"].fillna("Unnamed route")
#       df = df.dropna()
#       # expand list-like columns to percentage columns
#       surf_pct = df["surface"].apply(_parse_segment_percentages, mapping=surface_map)
#       way_pct = df["waytype"].apply(_parse_segment_percentages, mapping=waytype_map)
#       steep_pct = df["steepness"].apply(_parse_segment_percentages, mapping=steepness_map)
#       df = pd.concat([df.drop(columns=["surface", "waytype", "steepness", "waycategory"]),
#                       surf_pct.apply(pd.Series).fillna(0),
#                       way_pct.apply(pd.Series).fillna(0),
#                       steep_pct.apply(pd.Series).fillna(0)], axis=1)
#       df.to_csv(out_path, index=False)

#   def feature_engineer(in_path: str, out_path: str):
#       df = pd.read_csv(in_path)
#       df["Turn_Density"] = df["turns"] / (df["distance_m"] / 1000)
#       # surface buckets
#       df["on_road"] = df["Asphalt"] + df["Concrete"]; df.drop(columns=["Asphalt", "Concrete"], inplace=True)
#       df["off_road"] = df[["Dirt","Grass","Sand","Ground","Unpaved"]].sum(axis=1);
#   df.drop(columns=["Dirt","Grass","Sand","Ground","Unpaved"], inplace=True)
#       df["Gravel_Tracks"] = df["Gravel"] + df["Compacted Gravel"]; df.drop(columns=["Gravel","Compacted Gravel"], inplace=True)
#       df["Paved_Paths"] = df["Paved"] + df["Paving Stones"]; df.drop(columns=["Paved","Paving Stones"], inplace=True)
#       df["Other"] = df["Wood"] + df["Metal"] + df["Grass Paver"]; df.drop(columns=["Wood","Metal","Grass Paver"], inplace=True)
#       df["Unknown Surface"] = df.pop("Unknown")
#       # waytype buckets
#       df["Paved_Road"] = df["Road"] + df["Street"]; df.drop(columns=["Road","Street"], inplace=True)
#       df["Pedestrian"] = df["Steps"] + df["Footway"]; df.drop(columns=["Steps","Footway"], inplace=True)
#       df["Unknown_Way"] = df.pop("Unknown.1")
#       df["Cycle Track"] = df["Path"] + df["Track"]; df.drop(columns=["Path","Track"], inplace=True)
#       df["Other"] += df["Ferry"] + df["Construction"]; df.drop(columns=["Ferry","Construction"], inplace=True)
#       df["Main Road"] = df.pop("State Road")
#       # steepness buckets
#       df["Steep Section"] = df[["uphill_steep (5% to 7%)","uphill_very_steep (7% to 10%)","uphill_extreme (>10%)"]].sum(axis=1);
#   df.drop(columns=["uphill_steep (5% to 7%)","uphill_very_steep (7% to 10%)","uphill_extreme (>10%)"], inplace=True)
#       df["Moderate Section"] = df[["uphill_gentle (0% to 3%)","uphill_moderate (3% to 5%)"]].sum(axis=1);
#   df.drop(columns=["uphill_gentle (0% to 3%)","uphill_moderate (3% to 5%)"], inplace=True)
#       df["Flat Section"] = df.pop("flat (0%)")
#       df["Downhill Section"] = df[["downhill_gentle (-5% to 0%)","downhill_moderate (-7% to -5%)"]].sum(axis=1);
#   df.drop(columns=["downhill_gentle (-5% to 0%)","downhill_moderate (-7% to -5%)"], inplace=True)
#       df["Steep Downhill Section"] = df[["downhill_steep (-10% to -7%)","downhill_very_steep (-15% to -10%)","downhill_extreme
#   (<-15%)"]].sum(axis=1); df.drop(columns=["downhill_steep (-10% to -7%)","downhill_very_steep (-15% to -10%)","downhill_extreme
#   (<-15%)"], inplace=True)
#       df.to_csv(out_path, index=False)

#   def pipeline(...):
#       tmp1 = fetch_routes(..., out_path=combined_csv)
#       tmp2 = clean_routes(tmp1, processed_csv)
#       feature_engineer(tmp2, engineered_csv)

#   if __name__ == "__main__":
#       # wire subcommands: fetch, clean, fe, pipeline
