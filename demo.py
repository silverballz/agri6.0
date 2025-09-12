#!/usr/bin/env python3
"""
Demonstration script for the Agricultural Monitoring Platform.
Shows how to use the Sentinel-2A parser and geospatial utilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from data_processing.sentinel2_parser import parse_sentinel2_safe
from data_processing.geospatial_utils import CoordinateTransformer, utm_zone_from_longitude


def main():
    """Demonstrate the core functionality."""
    print("🛰️  Agricultural Monitoring Platform Demo")
    print("=" * 50)
    
    # Check if sample SAFE directory exists
    safe_dir = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE")
    
    if not safe_dir.exists():
        print("❌ Sample SAFE directory not found!")
        print(f"   Expected: {safe_dir}")
        return
    
    print(f"📁 Processing SAFE directory: {safe_dir.name}")
    
    try:
        # Parse Sentinel-2A data
        target_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']  # Key bands for vegetation analysis
        metadata, band_files = parse_sentinel2_safe(safe_dir, target_bands)
        
        print("\n📊 Metadata Information:")
        print(f"   Product ID: {metadata.product_id}")
        print(f"   Tile ID: {metadata.tile_id}")
        print(f"   Acquisition Date: {metadata.acquisition_date}")
        print(f"   Spacecraft: {metadata.spacecraft_name}")
        print(f"   Processing Level: {metadata.processing_level}")
        print(f"   Cloud Coverage: {metadata.cloud_coverage:.2f}%")
        print(f"   CRS: {metadata.epsg_code}")
        
        print(f"\n🎯 Found {len(band_files)} spectral bands:")
        for band_id, band_info in band_files.items():
            print(f"   {band_id}: {band_info.resolution} resolution, "
                  f"{band_info.central_wavelength}nm, "
                  f"File: {band_info.file_path.name}")
        
        # Demonstrate coordinate transformation
        print(f"\n🌍 Coordinate Transformation Demo:")
        
        # Calculate UTM zone from a longitude in the tile area
        sample_lon = 75.0  # Approximate longitude for T43REQ
        utm_zone = utm_zone_from_longitude(sample_lon)
        print(f"   Longitude {sample_lon}° → UTM Zone {utm_zone}")
        
        # Transform coordinates
        transformer = CoordinateTransformer('EPSG:4326', metadata.epsg_code)
        sample_lat = 25.0  # Approximate latitude for T43REQ
        utm_x, utm_y = transformer.transform_point(sample_lon, sample_lat)
        print(f"   WGS84 ({sample_lon}°, {sample_lat}°) → UTM ({utm_x:.0f}, {utm_y:.0f})")
        
        # Show vegetation index bands available
        print(f"\n🌱 Vegetation Index Calculation Ready:")
        vegetation_bands = {
            'NDVI': 'B08 (NIR) and B04 (Red)',
            'SAVI': 'B08 (NIR) and B04 (Red)',
            'EVI': 'B08 (NIR), B04 (Red), B02 (Blue)',
            'NDWI': 'B03 (Green) and B08 (NIR)',
            'NDSI': 'B11 (SWIR1) and B12 (SWIR2)'
        }
        
        for index_name, bands_needed in vegetation_bands.items():
            available_bands = [b for b in bands_needed.split(' and ') if any(b.startswith(band) for band in band_files.keys())]
            status = "✅" if len(available_bands) >= 2 or index_name == 'NDSI' else "⚠️"
            print(f"   {status} {index_name}: {bands_needed}")
        
        print(f"\n✅ Successfully processed Sentinel-2A data!")
        print(f"   Ready for vegetation index calculation and analysis.")
        
    except Exception as e:
        print(f"❌ Error processing SAFE directory: {e}")
        return


if __name__ == "__main__":
    main()