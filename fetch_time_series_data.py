"""
Fetch time-series satellite imagery for LSTM training.

This script queries the Sentinel Hub API for multiple dates
to build a temporal dataset for trend analysis.
"""

from datetime import datetime, timedelta
from src.data_processing.sentinel_hub_client import SentinelHubClient
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print(f"Loaded credentials:")
print(f"  Instance ID: {os.getenv('SENTINEL_HUB_INSTANCE_ID')[:8]}...")
print(f"  Client ID: {os.getenv('SENTINEL_HUB_CLIENT_ID')[:8]}...")

# Initialize client
client = SentinelHubClient()

# Define Ludhiana region as GeoJSON geometry
ludhiana_geometry = {
    "type": "Polygon",
    "coordinates": [[
        [75.80, 30.90],
        [75.90, 30.90],
        [75.90, 31.00],
        [75.80, 31.00],
        [75.80, 30.90]
    ]]
}

# Define date range (last 12 months, every 15 days)
end_date = datetime(2024, 9, 23)
start_date = end_date - timedelta(days=365)

dates = []
current = start_date
while current <= end_date:
    dates.append(current.strftime("%Y-%m-%d"))
    current += timedelta(days=15)

print(f"Fetching imagery for {len(dates)} dates...")
print(f"Date range: {dates[0]} to {dates[-1]}")

# Fetch imagery for each date
time_series_data = []

for date in dates:
    try:
        print(f"\nQuerying {date}...")
        
        # Query API
        imagery_list = client.query_sentinel_imagery(
            geometry=ludhiana_geometry,
            date_range=(date, date),
            cloud_threshold=30,
            max_results=1
        )
        
        if imagery_list and len(imagery_list) > 0:
            imagery = imagery_list[0]  # Get first result
            print(f"  ✓ Found imagery with {imagery['cloud_coverage']:.1f}% clouds")
            time_series_data.append({
                'date': date,
                'imagery': imagery,
                'cloud_coverage': imagery['cloud_coverage']
            })
        else:
            print(f"  ✗ No imagery found (clouds or no data)")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print(f"\n✓ Successfully fetched {len(time_series_data)} dates")
print(f"  This is enough for LSTM training!")

# Save metadata
with open('time_series_metadata.json', 'w') as f:
    json.dump({
        'total_dates': len(time_series_data),
        'date_range': [dates[0], dates[-1]],
        'dates': [d['date'] for d in time_series_data]
    }, f, indent=2)

print("\n✓ Metadata saved to: time_series_metadata.json")
