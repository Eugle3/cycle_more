import os
import pandas as pd

try:
    import google.generativeai as genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


MODEL = "gemini-2.0-flash"


def change_route_distance(df: pd.DataFrame, route_id, multiplier: float):
    """Return a copy of the route with distance and duration scaled."""
    rid = str(route_id)
    route = df.loc[df["id"].astype(str) == rid]
    if route.empty:
        return route
    return route.assign(
        distance_m=route["distance_m"] * float(multiplier),
        duration_s=route["duration_s"] * float(multiplier),
    )


# Tool schema from the notebook's Version 2 example
change_route_distance_tool = {
    "name": "change_route_distance",
    "description": "Return the route with its distance multiplied by the given factor (other features unchanged).",
    "parameters": {
        "type": "object",
        "properties": {
            "route_id": {"type": "string", "description": "The ID of the route to change."},
            "multiplier": {"type": "number", "description": "Factor to multiply the route's distance by."},
        },
        "required": ["route_id", "multiplier"],
    },
}

distance_tools = None if types is None else types.Tool(function_declarations=[change_route_distance_tool])


def run_distance_change_query(df: pd.DataFrame, user_question: str):
    """
    Ask Gemini to pick a route id and multiplier, then return the modified route copy.

    Returns a tuple: (route_df, tool_args, error_message)
    """
    if genai is None or types is None:
        return None, None, "google-generativeai/google-genai not installed"

    api_key = os.getenv("GEMINIKEY")
    if not api_key:
        return None, None, "Missing GEMINIKEY environment variable"

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=MODEL,
        contents=user_question,
        config=types.GenerateContentConfig(tools=[distance_tools]),
    )
    fc = resp.candidates[0].content.parts[0].function_call
    args = fc.args
    route = change_route_distance(df, args["route_id"], args["multiplier"])
    return route, args, None
