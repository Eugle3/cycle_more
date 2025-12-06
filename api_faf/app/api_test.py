from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .recommender import recommend_similar_routes


BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "Data_Engineered.csv")
feature_cols = df.drop(["id", "name"], axis=1).columns.tolist()
scaler = ColumnTransformer(
    transformers=[
        (
            "standard",
            StandardScaler(),
            [
                "distance_m",
                "duration_s",
                "ascent_m",
                "descent_m",
                "Turn_Density",
                "steps",
                "turns",
            ],
        ),
        (
            "minmax",
            MinMaxScaler(),
            [
                "Cycleway",
                "on_road",
                "off_road",
                "Gravel_Tracks",
                "Paved_Paths",
                "Other",
                "Unknown Surface",
                "Paved_Road",
                "Pedestrian",
                "Unknown_Way",
                "Cycle Track",
                "Main Road",
                "Steep Section",
                "Moderate Section",
                "Flat Section",
                "Downhill Section",
                "Steep Downhill Section",
            ],
        ),
    ],
    remainder="passthrough",
)
scaler.fit(df[feature_cols])
model = joblib.load(BASE_DIR / "model.pkl")

app = FastAPI()


@app.get("/")
def index() -> Dict[str, str]:
    return {"status": "ready"}


@app.post("/recommend")
def recommend_route(payload: dict = Body(...), n: int = 5):
    try:
        recs = recommend_similar_routes(
            payload, model, scaler, df, feature_cols, n
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"recommendations": recs}
