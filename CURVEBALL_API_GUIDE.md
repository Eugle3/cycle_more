# Curveball Recommendations API Guide

## Overview

Your API now has a new endpoint `/recommend-with-curveball` that returns:
- **5 similar routes** (KNN recommendations)
- **1 curveball route** from a different cluster type
- **Cluster labels** to explain the recommendation

## What Changed

### New Files
- `cycle_more/api_faf/app/kmeans.pkl` - K-means clustering model
- `cycle_more/api_faf/app/cluster_labels.py` - Human-readable cluster labels

### Updated Files
- `services.py` - Added k-means loading and curveball logic
- `main.py` - Added `/recommend-with-curveball` endpoint

## API Endpoints

### 1. POST `/recommend-with-curveball`

**Request:**
```json
{
  "features": {
    "distance_m": 5000,
    "duration_s": 1000,
    "ascent_m": 150,
    "descent_m": 150,
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
    "Steep Section": 5,
    "Moderate Section": 20,
    "Flat Section": 70,
    "Downhill Section": 3,
    "Steep Downhill Section": 2
  },
  "n_similar": 5
}
```

**Response:**
```json
{
  "similar": [
    {
      "route_id": 12345,
      "route_name": "City Loop",
      "distance_m": 5200,
      "ascent_m": 145,
      "duration_s": 1050,
      "turn_density": 1.5,
      "similarity_score": 0.05,
      "primary_surface": "Paved Road"
    },
    // ... 4 more similar routes
  ],
  "curveball": {
    "route_id": 67890,
    "route_name": "Alpine Challenge",
    "distance_m": 45000,
    "ascent_m": 1800,
    "duration_s": 9000,
    "turn_density": 0.8,
    "similarity_score": 2.5,
    "primary_surface": "Mixed"
  },
  "user_cluster_id": 0,
  "user_cluster_label": "Short and flat",
  "curveball_cluster_id": 9,
  "curveball_cluster_label": "Exceptionally long alpine steep"
}
```

### 2. Existing Endpoints (Still Work)

- `POST /recommend` - Original KNN recommendations (no curveball)
- `POST /recommend-from-gpx` - Upload GPX, get KNN recommendations

## Testing Locally

### 1. Start the API

```bash
cd cycle_more/api_faf
uvicorn app.main:app --reload
```

### 2. Test with curl

```bash
curl -X POST "http://localhost:8000/recommend-with-curveball" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "distance_m": 5000,
      "duration_s": 1000,
      "ascent_m": 150,
      "descent_m": 150,
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
      "Steep Section": 5,
      "Moderate Section": 20,
      "Flat Section": 70,
      "Downhill Section": 3,
      "Steep Downhill Section": 2
    },
    "n_similar": 5
  }'
```

### 3. Test with Python

```python
import requests

url = "http://localhost:8000/recommend-with-curveball"

payload = {
    "features": {
        "distance_m": 5000,
        "duration_s": 1000,
        "ascent_m": 150,
        "descent_m": 150,
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
        "Steep Section": 5,
        "Moderate Section": 20,
        "Flat Section": 70,
        "Downhill Section": 3,
        "Steep Downhill Section": 2
    },
    "n_similar": 5
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Your route is: {result['user_cluster_label']}")
print(f"\nSimilar routes ({len(result['similar'])}):")
for route in result['similar']:
    print(f"  - {route['route_name']}: {route['distance_m']/1000:.1f}km, {route['ascent_m']:.0f}m")

print(f"\nCurveball ({result['curveball_cluster_label']}):")
cb = result['curveball']
print(f"  - {cb['route_name']}: {cb['distance_m']/1000:.1f}km, {cb['ascent_m']:.0f}m")
```

## Frontend Integration

### Display Example

```javascript
// After calling the API
const result = await fetch('/recommend-with-curveball', {
  method: 'POST',
  body: JSON.stringify({ features: userFeatures, n_similar: 5 })
}).then(r => r.json());

// Display to user
console.log(`Your route is a "${result.user_cluster_label}"`);
console.log(`\nRoutes similar to yours:`);
result.similar.forEach(route => {
  console.log(`  - ${route.route_name}`);
});

console.log(`\nTry something different!`);
console.log(`  Curveball: ${result.curveball.route_name}`);
console.log(`  (This is a "${result.curveball_cluster_label}")`);
```

## Cluster Labels

Your current clusters (from `cluster_labels.py`):
- 0: "Short and flat"
- 1: "Very long alpine mixed"
- 2: "Very short and flat"
- 3: "Mixed surface multi-day tours"
- 4: "Mid-length rolling hills"
- 5: "Very Short Flat Mixed"
- 6: "Long alpine mixed"
- 7: "Very long rolling hills"
- 8: "Very short flat mixed"
- 9: "Exceptionally long alpine steep"

## Deployment

When deploying to Google Cloud Run, make sure:
1. ✅ `kmeans.pkl` is in the Docker image at `app/kmeans.pkl`
2. ✅ `cluster_labels.py` is in the Docker image at `app/cluster_labels.py`
3. ✅ No changes needed to Dockerfile (files are in the app folder)

## Future Improvements

### Option 1: True Cluster Filtering
Currently, the curveball is selected from the top 100 nearest neighbors (excluding the 5 similar routes). To improve this:

1. Add cluster column to `Data_Engineered.csv`:
   ```python
   # In your notebook after clustering
   df.to_csv('Data_Engineered.csv', index=False)  # Include cluster column
   ```

2. Update `services.py` to filter by actual cluster:
   ```python
   # In recommend_with_curveball()
   curveball_routes = df[df['cluster'] == curveball_cluster_id]
   # Then find nearest from this subset
   ```

### Option 2: Smart Curveball Selection
Instead of random cluster, select the "most different" cluster based on centroid distance.

### Option 3: GPX Upload with Curveball
Add a new endpoint `/recommend-from-gpx-with-curveball` that combines GPX processing with curveball recommendations.

## Troubleshooting

### Import Error: "No module named 'cluster_labels'"
- Make sure `cluster_labels.py` is in `cycle_more/api_faf/app/`

### Kmeans.pkl not found
- Check that `kmeans.pkl` was copied to `cycle_more/api_faf/app/`
- File size should be ~99KB

### All recommendations are similar (no true curveball)
- This is expected in v1 - we're not filtering by cluster yet
- See "Future Improvements" above to add true cluster filtering
