from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .services import (
    change_route_distance,
    load_data,
    load_model_and_scaler,
    load_kmeans,
    recommend_routes,
    recommend_with_curveball,
    recommend_from_prompt,
    predict_cluster,
    run_distance_change_query,
    process_gpx_upload,
    process_gpx_upload_with_curveball,
)
from .gpx_generator import generate_gpx_for_route
from .route_visualizer import visualize_route


app = FastAPI(title="CycleMore API", version="0.1.0")

# Warm caches at startup so the first request is fast.
_df, _feature_cols = load_data()
_model, _scaler = load_model_and_scaler()
_kmeans = load_kmeans()


class RecommendRequest(BaseModel):
    features: Dict[str, Any]
    n_recommendations: int = Field(default=5, ge=1, le=20)


class Recommendation(BaseModel):
    route_id: int
    route_name: str
    distance_m: float
    ascent_m: float
    duration_s: float
    turn_density: float
    similarity_score: float
    primary_surface: str


class DistanceChangeRequest(BaseModel):
    route_id: str
    multiplier: float = Field(..., gt=0)
    use_llm: bool = False
    prompt: Optional[str] = None


class CurveballRequest(BaseModel):
    features: Dict[str, Any]
    n_similar: int = Field(default=5, ge=1, le=20)


class CurveballResponse(BaseModel):
    similar: List[Recommendation]
    curveball: Optional[Recommendation]
    user_cluster_id: int
    user_cluster_label: str
    curveball_cluster_id: int
    curveball_cluster_label: str


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Natural language description of desired route")
    n_similar: int = Field(default=5, ge=1, le=20)


class PromptResponse(BaseModel):
    similar: List[Recommendation]
    curveball: Optional[Recommendation]
    user_cluster_id: int
    user_cluster_label: str
    curveball_cluster_id: int
    curveball_cluster_label: str
    generated_features: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=List[Recommendation])
def recommend(req: RecommendRequest):
    try:
        recs = recommend_routes(req.features, n_recommendations=req.n_recommendations)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return recs


@app.post("/recommend-with-curveball", response_model=CurveballResponse)
def recommend_curveball(req: CurveballRequest):
    """
    Get route recommendations including a "curveball" from a different cluster.

    Returns 5 similar routes (KNN) plus 1 curveball route from a different cluster
    to give users variety and help them discover new types of routes.

    The response includes cluster labels to help explain the curveball:
    - "Your route is a 'Short flat city ride'"
    - "Try this 'Mountain climbing challenge' for something different!"
    """
    try:
        result = recommend_with_curveball(req.features, n_similar=req.n_similar)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(exc)}")
    return result


@app.post("/recommend-from-prompt", response_model=PromptResponse)
def recommend_from_prompt_endpoint(req: PromptRequest):
    """
    Generate route recommendations from a natural language prompt.

    This endpoint uses LLM (GPT) to convert your description into route features,
    then finds similar routes using the KNN model plus a curveball from a different cluster.

    Examples:
    - "A flat 10 km loop around Richmond Park, mostly paved, low traffic"
    - "A challenging mountain route with steep climbs and gravel sections"
    - "An easy 5km urban cycle path suitable for beginners"

    The response includes:
    - Similar routes matching your description
    - A curveball route from a different cluster for variety
    - The generated features (for debugging/transparency)
    """
    try:
        result = recommend_from_prompt(
            user_prompt=req.prompt,
            n_similar=req.n_similar
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(exc)}")
    return result


@app.post("/distance-change")
def distance_change(req: DistanceChangeRequest):
    df: pd.DataFrame = _df

    if req.use_llm:
        prompt = req.prompt or f"Route id: {req.route_id}. Make it longer."
        changed, tool_args, err = run_distance_change_query(df, prompt)
        if err:
            raise HTTPException(status_code=503, detail=err)
    else:
        changed = change_route_distance(df, req.route_id, req.multiplier)
        tool_args = {"route_id": req.route_id, "multiplier": req.multiplier}

    if changed is None or changed.empty:
        raise HTTPException(status_code=404, detail="Route not found")

    row = changed.iloc[0].to_dict()
    return {"route": row, "tool_args": tool_args}


@app.post("/recommend-from-gpx", response_model=CurveballResponse)
async def recommend_from_gpx(file: UploadFile = File(...)):
    """
    Upload a GPX file and get route recommendations with a curveball.

    The GPX file is processed through:
    1. Coordinate extraction (smart sampling to max 70 waypoints)
    2. ORS API call to get route features
    3. Feature engineering (same as training data)
    4. KNN model prediction with curveball from different cluster

    Returns:
        CurveballResponse with similar routes and a curveball recommendation
    """
    # Validate file type
    if not file.filename.endswith('.gpx'):
        raise HTTPException(status_code=400, detail="File must be a GPX file")

    try:
        # Read GPX content
        gpx_content = await file.read()

        # Process GPX → features → recommendations with curveball
        result = process_gpx_upload_with_curveball(gpx_content)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"GPX parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/download-gpx/{route_id}")
def download_gpx(route_id: int):
    """
    Download a route as a GPX file.

    Args:
        route_id: The route ID to download

    Returns:
        GPX file ready for download
    """
    # Get route name from database
    df, _ = load_data()
    route_row = df[df['id'] == route_id]

    if route_row.empty:
        raise HTTPException(status_code=404, detail=f"Route {route_id} not found in database")

    route_name = str(route_row.iloc[0]['name'])

    try:
        # Generate GPX from GCS
        gpx_xml = generate_gpx_for_route(
            route_id=route_id,
            route_name=route_name,
            bucket="cycle_more_bucket",
            prefix="all_routes/"
        )

        # Return as downloadable file
        return Response(
            content=gpx_xml,
            media_type="application/gpx+xml",
            headers={
                "Content-Disposition": f'attachment; filename="route_{route_id}.gpx"'
            }
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Route data not found in GCS: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating GPX: {str(e)}"
        )


@app.get("/visualize-route/{route_id}")
def visualize_route_endpoint(route_id: int):
    """
    Get interactive map visualization HTML for a route.

    Args:
        route_id: The route ID to visualize

    Returns:
        HTML content of the Folium map
    """
    # Get route details from database
    df, _ = load_data()
    route_row = df[df['id'] == route_id]

    if route_row.empty:
        raise HTTPException(status_code=404, detail=f"Route {route_id} not found in database")

    route_name = str(route_row.iloc[0]['name'])
    distance_km = float(route_row.iloc[0]['distance_m']) / 1000
    ascent_m = float(route_row.iloc[0]['ascent_m'])

    try:
        # Generate Folium map HTML
        map_html = visualize_route(
            route_id=route_id,
            route_name=route_name,
            distance_km=distance_km,
            ascent_m=ascent_m,
            bucket="cycle_more_bucket"
        )

        # Return HTML
        return Response(
            content=map_html,
            media_type="text/html"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Route data not found in GCS: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid route data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visualization: {str(e)}"
        )
