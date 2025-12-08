import streamlit as st
import pandas as pd
import joblib
import requests
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from recommender import recommend_similar_routes
from llm_distance import change_route_distance as llm_change_route_distance, run_distance_change_query


BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "Data_Engineered.csv")
    # Drop non-feature columns (id, name, region)
    feature_cols = df.drop(["id", "name", "region"], axis=1).columns.tolist()
    return df, feature_cols

@st.cache_resource
def load_model_and_scaler(feature_cols, df):
    model = joblib.load(BASE_DIR / "model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler

df, feature_cols = load_data()
model, scaler = load_model_and_scaler(feature_cols, df)

st.title("CycleMore Route Recommender")

# Configuration for API endpoint
API_URL = "http://localhost:8000"

# ========================================
# NEW FEATURE: Upload Your Own GPX File
# ========================================
st.header("🚴 Upload Your Own Ride")
st.markdown("Upload a GPX file from your bike computer or tracking app to find similar routes!")

uploaded_file = st.file_uploader("Choose a GPX file", type=['gpx'])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Find Similar Routes", key="gpx_recommend"):
        with st.spinner("Processing your route..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/gpx+xml")}

                # Call the FastAPI endpoint
                response = requests.post(f"{API_URL}/recommend-from-gpx", files=files)

                if response.status_code == 200:
                    recommendations = response.json()

                    st.success(f"Found {len(recommendations)} similar routes!")

                    # Display recommendations
                    st.subheader("Recommended Routes")
                    recs_df = pd.DataFrame(recommendations)

                    # Format the dataframe for better display
                    display_df = recs_df.copy()
                    display_df['distance_km'] = (display_df['distance_m'] / 1000).round(2)
                    display_df['duration_min'] = (display_df['duration_s'] / 60).round(1)
                    display_df['ascent_m'] = display_df['ascent_m'].round(0)

                    # Select and rename columns for display
                    display_df = display_df[[
                        'route_name', 'distance_km', 'ascent_m',
                        'duration_min', 'turn_density', 'similarity_score'
                    ]]
                    display_df.columns = [
                        'Route Name', 'Distance (km)', 'Ascent (m)',
                        'Duration (min)', 'Turn Density', 'Similarity Score'
                    ]

                    st.dataframe(display_df, use_container_width=True)

                else:
                    st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to API. Make sure FastAPI is running on http://localhost:8000")
                st.info("Run: `uvicorn app.main:app --reload` from the api_faf directory")
            except Exception as e:
                st.error(f"Error processing GPX file: {str(e)}")

st.divider()

# ========================================
# EXISTING FEATURES: Browse & Modify Routes
# ========================================
st.header("📋 Browse Existing Routes")

route_name = st.selectbox("Pick a route to find similar", options=df["name"].unique())
seed_row = df.loc[df["name"] == route_name].iloc[0]
if st.session_state.get("current_route_id") != seed_row["id"]:
    st.session_state["current_route"] = seed_row
    st.session_state["current_route_id"] = seed_row["id"]
    st.session_state["current_payload"] = seed_row[feature_cols].to_dict()

st.subheader("Make this route longer (LLM or fallback)")
prompt = st.text_input("What do you want?", value="Like this route but longer")
multiplier = st.slider("Fallback distance multiplier", 1.0, 3.0, 2.0, 0.1, key="fallback_multiplier")
use_llm = st.checkbox("Use LLM (Gemini) for multiplier/id selection", value=True)

if st.button("Generate longer route"):
    if use_llm:
        user_question = f"Route id: {seed_row['id']}. {prompt}"
        changed, tool_args, err = run_distance_change_query(df, user_question)
        if err:
            st.warning(f"LLM unavailable: {err}. Falling back to local multiplier.")
            changed = llm_change_route_distance(df, seed_row["id"], multiplier)
            tool_args = {"route_id": seed_row["id"], "multiplier": multiplier}
    else:
        changed = llm_change_route_distance(df, seed_row["id"], multiplier)
        tool_args = {"route_id": seed_row["id"], "multiplier": multiplier}

    if changed is None or changed.empty:
        st.warning("Could not find that route.")
    else:
        st.session_state["current_route"] = changed.iloc[0]
        st.session_state["current_route_id"] = changed.iloc[0]["id"]
        st.session_state["current_payload"] = changed[feature_cols].iloc[0].to_dict()
        st.caption(f"Tool args: {tool_args}")
        st.dataframe(changed)

n_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if st.button("Recommend based on current route"):
    route_for_recs = st.session_state.get("current_route", seed_row)
    payload = st.session_state.get("current_payload", seed_row[feature_cols].to_dict())

    recs = recommend_similar_routes(
        input_features=payload,
        model=model,
        scaler=scaler,
        df=df,
        feature_cols=feature_cols,
        n_recommendations=n_recs,
    )

    # Drop the identical route if it appears as its own nearest neighbor
    filtered = [r for r in recs if r["route_id"] != int(route_for_recs["id"])]

    st.subheader("Recommendations")
    st.dataframe(pd.DataFrame(filtered) if filtered else pd.DataFrame(recs))
