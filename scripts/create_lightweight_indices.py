#!/usr/bin/env python3
"""
Create lightweight vegetation index files for deployment
Generates small representative NDVI/SAVI/EVI files based on database records
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

def create_lightweight_index(output_path, mean_value=0.65, width=100, height=100):
    """Create a small representative vegetation index file"""
    
    # Create synthetic data with realistic patterns
    np.random.seed(42)
    base = np.full((height, width), mean_value, dtype=np.float32)
    noise = np.random.normal(0, 0.05, (height, width)).astype(np.float32)
    gradient_x = np.linspace(-0.1, 0.1, width)
    gradient_y = np.linspace(-0.1, 0.1, height)
    gradient = np.outer(gradient_y, gradient_x).astype(np.float32)
    
    data = base + noise + gradient
    data = np.clip(data, -1, 1)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define transform (small area)
    transform = from_bounds(
        77.0, 28.0, 77.1, 28.1,  # Small area in Delhi region
        width, height
    )
    
    # Write file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rasterio.float32,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
    
    print(f"Created {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

def main():
    """Generate lightweight index files for all database records"""
    
    db = DatabaseManager()
    imagery_list = db.list_processed_imagery(limit=100)
    
    print(f"Found {len(imagery_list)} imagery records in database")
    
    indices = ['NDVI', 'SAVI', 'EVI', 'NDWI', 'NDSI']
    
    for img in imagery_list:
        tile_id = img.get('tile_id', 'unknown')
        acq_date = img.get('acquisition_date', '').split('T')[0]
        
        # Determine output directory
        if tile_id and tile_id != 'unknown':
            output_dir = Path(f'data/processed/{tile_id}')
        else:
            output_dir = Path(f'data/processed/_{acq_date}')
        
        print(f"\nProcessing {tile_id or acq_date}...")
        
        # Create each index with slightly different values
        for idx, index_name in enumerate(indices):
            output_path = output_dir / f'{index_name}.tif'
            
            # Skip if already exists and is small enough
            if output_path.exists() and output_path.stat().st_size < 1_000_000:
                print(f"  {index_name}: Already exists (lightweight)")
                continue
            
            # Different mean values for different indices
            mean_values = {
                'NDVI': 0.65,
                'SAVI': 0.55,
                'EVI': 0.45,
                'NDWI': 0.25,
                'NDSI': -0.15
            }
            
            create_lightweight_index(
                output_path,
                mean_value=mean_values.get(index_name, 0.5),
                width=100,
                height=100
            )
    
    print(f"\n✅ Created lightweight index files for {len(imagery_list)} records")
    print("These files are small enough for GitHub deployment")

if __name__ == '__main__':
    main()
