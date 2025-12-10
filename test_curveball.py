"""
Quick test script for curveball recommendations.
Run with: python test_curveball.py
"""

import sys
sys.path.insert(0, 'api_faf')

from app.services import recommend_with_curveball, load_data

# Load data to get a sample route
df, _ = load_data()

# Get features from an existing route (first route)
sample_route = df.iloc[100]
print(f"Testing with route: {sample_route['name']}")
print(f"  Distance: {sample_route['distance_m']/1000:.1f}km")
print(f"  Ascent: {sample_route['ascent_m']:.0f}m")
print(f"  Cluster: {sample_route['cluster']}")
print()

# Get curveball recommendations
features = sample_route.drop(['id', 'name', 'region', 'cluster']).to_dict()
result = recommend_with_curveball(features, n_similar=5)

print("=" * 80)
print(f"Your route is classified as: '{result['user_cluster_label']}' (Cluster {result['user_cluster_id']})")
print("=" * 80)

print(f"\n📍 SIMILAR ROUTES ({len(result['similar'])}):")
for i, route in enumerate(result['similar'], 1):
    print(f"  {i}. {route['route_name'][:50]:50s} | {route['distance_m']/1000:5.1f}km | {route['ascent_m']:5.0f}m")

print(f"\n🎲 CURVEBALL ('{result['curveball_cluster_label']}' - Cluster {result['curveball_cluster_id']}):")
cb = result['curveball']
print(f"  → {cb['route_name'][:50]:50s} | {cb['distance_m']/1000:5.1f}km | {cb['ascent_m']:5.0f}m")
print(f"  → Similarity score: {cb['similarity_score']:.4f}")

print("\n✅ Test completed successfully!")
