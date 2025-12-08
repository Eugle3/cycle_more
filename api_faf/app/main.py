from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .services import (
    change_route_distance,
    load_data,
    load_model_and_scaler,
    recommend_routes,
    run_distance_change_query,
)


app = FastAPI(title="CycleMore API", version="0.1.0")

# Warm caches at startup so the first request is fast.
_df, _feature_cols = load_data()
_model, _scaler = load_model_and_scaler()


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


class DistanceChangeRequest(BaseModel):
    route_id: str
    multiplier: float = Field(..., gt=0)
    use_llm: bool = False
    prompt: Optional[str] = None


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
