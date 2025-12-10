"""
LLM-based feature generation from natural language prompts.

This module uses OpenAI's GPT models to convert user descriptions
(e.g., "A flat 10 km loop around Richmond Park") into the 27 features
needed by the KNN route recommendation model.
"""

import os
import csv
import io
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Column names matching the training data
COLUMNS = [
    "id", "name", "distance_m", "duration_s", "ascent_m", "descent_m", "steps", "turns",
    "region", "Cycleway", "Turn_Density", "on_road", "off_road", "Gravel_Tracks",
    "Paved_Paths", "Other", "Unknown Surface", "Paved_Road", "Pedestrian", "Unknown_Way",
    "Cycle Track", "Main Road", "Steep Section", "Moderate Section", "Flat Section",
    "Downhill Section", "Steep Downhill Section"
]

# Dataset mean values as fallback defaults (from df.describe())
DEFAULTS = {
    "distance_m": 5421.8072,
    "duration_s": 1126.8549,
    "ascent_m": 69.5386,
    "descent_m": 67.6627,
    "steps": 10.029953,
    "turns": 6.741557,
    "Cycleway": 20.634499,
    "Turn_Density": 2.653734,
    "on_road": 66.504514,
    "off_road": 0.901243,
    "Gravel_Tracks": 3.495492,
    "Paved_Paths": 5.377312,
    "Other": 0.092803,
    "Unknown Surface": 23.583983,
    "Paved_Road": 59.350565,
    "Pedestrian": 2.927223,
    "Unknown_Way": 0.214249,
    "Cycle Track": 11.988817,
    "Main Road": 4.791810,
    "Steep Section": 1.018174,
    "Moderate Section": 18.202830,
    "Flat Section": 57.723090,
    "Downhill Section": 19.178338,
    "Steep Downhill Section": 0.887704,
}

SYSTEM_PROMPT = """
You are a route-row generator. Return exactly one CSV row including header.
Columns (27): id,name,distance_m,duration_s,ascent_m,descent_m,steps,turns,region,
Cycleway,Turn_Density,on_road,off_road,Gravel_Tracks,Paved_Paths,Other,Unknown Surface,
Paved_Road,Pedestrian,Unknown_Way,Cycle Track,Main Road,Steep Section,Moderate Section,
Flat Section,Downhill Section,Steep Downhill Section

Rules:
- id must be "Synthetic_Route_1".
- name is a short descriptive title.
- region is a short category.
- distance_m, duration_s, ascent_m, descent_m, steps, turns are non-negative numbers.
- The remaining fields are percentages/fractions; keep each between 0 and 100 and sum logically.

If unsure, use the typical defaults below.
Typical defaults (use if unspecified): distance_m=5421.8072, duration_s=1126.8549,
ascent_m=69.5386, descent_m=67.6627, steps=10.029953, turns=6.741557,
Cycleway=20.634499, Turn_Density=2.653734, on_road=66.504514, off_road=0.901243,
Gravel_Tracks=3.495492, Paved_Paths=5.377312, Other=0.092803, Unknown Surface=23.583983,
Paved_Road=59.350565, Pedestrian=2.927223, Unknown_Way=0.214249, Cycle Track=11.988817,
Main Road=4.791810, Steep Section=1.018174, Moderate Section=18.202830,
Flat Section=57.723090, Downhill Section=19.178338, Steep Downhill Section=0.887704.

Output only CSV text with the header line and one data line. No extra text.
""".strip()


def clamp_pct(x: Any) -> float:
    """Clamp percentage values to [0, 100] range."""
    try:
        return max(0.0, min(float(x), 100.0))
    except (ValueError, TypeError):
        return 0.0


def clamp_positive(x: Any) -> float:
    """Clamp numeric values to be non-negative."""
    try:
        return max(0.0, float(x))
    except (ValueError, TypeError):
        return 0.0


def fill_and_clamp(row_dict: Dict[str, Any], region_default: str = "Unknown") -> Dict[str, Any]:
    """
    Fill missing values with defaults and clamp all values to valid ranges.

    Args:
        row_dict: Parsed CSV row from LLM
        region_default: Default region if not specified

    Returns:
        Dictionary with all 27 features properly filled and clamped
    """
    out = {}

    # Percentage columns (need clamping to 0-100)
    pct_cols = {
        "Cycleway", "on_road", "off_road", "Gravel_Tracks", "Paved_Paths", "Other",
        "Unknown Surface", "Paved_Road", "Pedestrian", "Unknown_Way", "Cycle Track",
        "Main Road", "Steep Section", "Moderate Section", "Flat Section",
        "Downhill Section", "Steep Downhill Section"
    }

    # Numeric columns (only need to be non-negative)
    numeric_cols = {
        "distance_m", "duration_s", "ascent_m", "descent_m", "steps", "turns", "Turn_Density"
    }

    for col in COLUMNS:
        if col == "id":
            out[col] = "Synthetic_Route_1"
        elif col == "name":
            out[col] = row_dict.get(col) or "Synthetic Route"
        elif col == "region":
            out[col] = row_dict.get(col) or region_default
        elif col in pct_cols:
            val = row_dict.get(col, DEFAULTS.get(col, 0))
            out[col] = clamp_pct(val)
        elif col in numeric_cols:
            val = row_dict.get(col, DEFAULTS.get(col, 0))
            out[col] = clamp_positive(val)
        else:
            out[col] = row_dict.get(col, 0)

    return out


def generate_features_from_prompt(
    user_prompt: str,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Generate route features from a natural language prompt using OpenAI.

    Args:
        user_prompt: Natural language description of desired route
                    e.g., "A flat 10 km loop around Richmond Park, mostly paved"
        openai_api_key: OpenAI API key (uses OPENKEY env var if not provided)
        model: OpenAI model to use (default: gpt-4o-mini)
        temperature: Model temperature (default: 0.2 for more consistent output)

    Returns:
        Dictionary of 27 route features ready for KNN model

    Raises:
        ValueError: If OpenAI library not installed or API key missing
        Exception: If LLM call fails or CSV parsing fails
    """
    if OpenAI is None:
        raise ValueError("openai library not installed. Run: pip install openai")

    api_key = openai_api_key or os.getenv("OPENKEY")
    if not api_key:
        raise ValueError("Missing OPENKEY environment variable or openai_api_key parameter")

    # Call OpenAI to generate CSV
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=300,
        )

        csv_text = response.choices[0].message.content

        # Parse CSV output
        reader = csv.DictReader(io.StringIO(csv_text))
        row = next(reader)

        # Fill missing values and clamp to valid ranges
        features = fill_and_clamp(row)

        return features

    except Exception as e:
        raise Exception(f"Failed to generate features from prompt: {str(e)}")
