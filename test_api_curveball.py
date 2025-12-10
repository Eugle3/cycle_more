"""
Test the /recommend-with-curveball API endpoint locally.
"""

import requests
import json

# API endpoint
url = "http://localhost:8000/recommend-with-curveball"

# Sample route features (short flat route)
payload = {
    "features": {
        "distance_m": 5000,
        "duration_s": 1000,
        "ascent_m": 50,
        "descent_m": 50,
        "steps": 2,
        "turns": 8,
        "Cycleway": 0,
        "Turn_Density": 1.6,
        "on_road": 90,
        "off_road": 10,
        "Gravel_Tracks": 0,
        "Paved_Paths": 10,
        "Other": 0,
        "Unknown Surface": 0,
        "Paved_Road": 90,
        "Pedestrian": 0,
        "Unknown_Way": 0,
        "Cycle Track": 0,
        "Main Road": 0,
        "Steep Section": 0,
        "Moderate Section": 10,
        "Flat Section": 85,
        "Downhill Section": 3,
        "Steep Downhill Section": 2
    },
    "n_similar": 5
}

print("🚀 Testing /recommend-with-curveball endpoint...")
print(f"URL: {url}")
print(f"\n📤 Request:")
print(f"  Distance: {payload['features']['distance_m']/1000:.1f}km")
print(f"  Ascent: {payload['features']['ascent_m']:.0f}m")
print(f"  On-road: {payload['features']['on_road']:.0f}%")
print(f"  Flat: {payload['features']['Flat Section']:.0f}%")

try:
    response = requests.post(url, json=payload, timeout=10)

    if response.status_code == 200:
        result = response.json()

        print("\n" + "=" * 80)
        print(f"✅ SUCCESS! Status: {response.status_code}")
        print("=" * 80)

        print(f"\n🏷️  YOUR ROUTE CLASSIFICATION:")
        print(f"  Cluster {result['user_cluster_id']}: '{result['user_cluster_label']}'")

        print(f"\n📍 SIMILAR ROUTES ({len(result['similar'])}):")
        for i, route in enumerate(result['similar'], 1):
            print(f"  {i}. {route['route_name'][:45]:45s} | {route['distance_m']/1000:5.1f}km | {route['ascent_m']:5.0f}m | Score: {route['similarity_score']:.3f}")

        print(f"\n🎲 CURVEBALL RECOMMENDATION:")
        print(f"  Cluster {result['curveball_cluster_id']}: '{result['curveball_cluster_label']}'")
        cb = result['curveball']
        print(f"  → {cb['route_name'][:45]:45s} | {cb['distance_m']/1000:5.1f}km | {cb['ascent_m']:5.0f}m")
        print(f"  → Similarity score: {cb['similarity_score']:.3f}")
        print(f"  → Surface: {cb['primary_surface']}")

        print("\n" + "=" * 80)
        print("💡 RECOMMENDATION MESSAGE:")
        print("=" * 80)
        print(f"Your route is a '{result['user_cluster_label']}'.")
        print(f"Try this '{result['curveball_cluster_label']}' for something different!")
        print()

    else:
        print(f"\n❌ ERROR: Status {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Could not connect to server. Is it running?")
    print("Start with: cd api_faf && uvicorn app.main:app --reload")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
