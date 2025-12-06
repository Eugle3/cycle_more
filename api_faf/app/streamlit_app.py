import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from recommender import recommend_similar_routes

BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "Data_Engineered.csv")
    feature_cols = df.drop(["id", "name"], axis=1).columns.tolist()
    return df, feature_cols

@st.cache_resource
def load_model_and_scaler(feature_cols, df):
    model = joblib.load(BASE_DIR / "model.pkl")
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
    return model, scaler

df, feature_cols = load_data()
model, scaler = load_model_and_scaler(feature_cols, df)

st.title("CycleMore Route Recommender (Minimal)")

route_name = st.selectbox("Pick a route to find similar", options=df["name"].unique())
n_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if st.button("Recommend"):
    seed_row = df.loc[df["name"] == route_name].iloc[0]
    payload = seed_row[feature_cols].to_dict()

    recs = recommend_similar_routes(
        input_features=payload,
        model=model,
        scaler=scaler,
        df=df,
        feature_cols=feature_cols,
        n_recommendations=n_recs,
    )

    st.subheader("Recommendations")
    st.dataframe(pd.DataFrame(recs))
