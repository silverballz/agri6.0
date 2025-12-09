"""
Visualize Sentinel-2A satellite imagery from the local data.

This script loads the 4 bands (B02, B03, B04, B08) and creates:
1. True color RGB composite
2. False color infrared composite (for vegetation)
3. NDVI visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from pathlib import Path

# Paths to the 10m resolution bands
base_path = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE/GRANULE/L2A_T43REQ_A048336_20240923T055000/IMG_DATA/R10m")

band_files = {
    'B02': base_path / "T43REQ_20240923T053641_B02_10m.jp2",  # Blue
    'B03': base_path / "T43REQ_20240923T053641_B03_10m.jp2",  # Green
    'B04': base_path / "T43REQ_20240923T053641_B04_10m.jp2",  # Red
    'B08': base_path / "T43REQ_20240923T053641_B08_10m.jp2",  # NIR
}

print("Loading Sentinel-2A bands...")
bands = {}
for band_name, file_path in band_files.items():
    with rasterio.open(file_path) as src:
        bands[band_name] = src.read(1).astype(np.float32)
        print(f"  {band_name}: {bands[band_name].shape}, range: {bands[band_name].min():.0f}-{bands[band_name].max():.0f}")

# Get a random crop for visualization (full image is 10980x10980, too large)
crop_size = 1000
h, w = bands['B02'].shape
start_y = np.random.randint(0, h - crop_size)
start_x = np.random.randint(0, w - crop_size)

print(f"\nCropping region: [{start_y}:{start_y+crop_size}, {start_x}:{start_x+crop_size}]")

# Crop all bands
for band_name in bands:
    bands[band_name] = bands[band_name][start_y:start_y+crop_size, start_x:start_x+crop_size]

# Normalize bands to 0-1 range for visualization
def normalize_band(band, percentile_clip=2):
    """Normalize band with percentile clipping for better contrast."""
    p_low = np.percentile(band, percentile_clip)
    p_high = np.percentile(band, 100 - percentile_clip)
    band_clipped = np.clip(band, p_low, p_high)
    return (band_clipped - p_low) / (p_high - p_low)

# Calculate NDVI
nir = bands['B08']
red = bands['B04']
ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# 1. True Color RGB (B04-Red, B03-Green, B02-Blue)
rgb = np.stack([
    normalize_band(bands['B04']),
    normalize_band(bands['B03']),
    normalize_band(bands['B02'])
], axis=-1)

axes[0, 0].imshow(rgb)
axes[0, 0].set_title('True Color RGB\n(Red, Green, Blue)', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# 2. False Color Infrared (B08-NIR, B04-Red, B03-Green)
# Vegetation appears red/pink in this composite
false_color = np.stack([
    normalize_band(bands['B08']),
    normalize_band(bands['B04']),
    normalize_band(bands['B03'])
], axis=-1)

axes[0, 1].imshow(false_color)
axes[0, 1].set_title('False Color Infrared\n(NIR, Red, Green) - Vegetation = Red/Pink', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# 3. NDVI with custom colormap
ndvi_cmap = LinearSegmentedColormap.from_list('ndvi', [
    '#8B4513',  # Brown (bare soil/low vegetation)
    '#FFFF00',  # Yellow
    '#90EE90',  # Light green
    '#228B22',  # Forest green (healthy vegetation)
    '#006400'   # Dark green (very healthy)
])

im = axes[1, 0].imshow(ndvi, cmap=ndvi_cmap, vmin=-0.2, vmax=0.9)
axes[1, 0].set_title('NDVI (Vegetation Index)\nGreen = Healthy Vegetation', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='NDVI Value')

# 4. Statistics and info
axes[1, 1].axis('off')
info_text = f"""
Sentinel-2A Satellite Image
Tile: 43REQ (Ludhiana Region, Punjab, India)
Date: September 23, 2024
Time: 05:36:41 UTC

Image Details:
• Resolution: 10 meters per pixel
• Crop size: {crop_size} × {crop_size} pixels
• Area covered: ~{crop_size*10/1000:.1f} × {crop_size*10/1000:.1f} km

Band Statistics (cropped region):
• B02 (Blue):  {bands['B02'].mean():.0f} ± {bands['B02'].std():.0f}
• B03 (Green): {bands['B03'].mean():.0f} ± {bands['B03'].std():.0f}
• B04 (Red):   {bands['B04'].mean():.0f} ± {bands['B04'].std():.0f}
• B08 (NIR):   {bands['B08'].mean():.0f} ± {bands['B08'].std():.0f}

NDVI Statistics:
• Mean: {ndvi.mean():.3f}
• Std:  {ndvi.std():.3f}
• Min:  {ndvi.min():.3f}
• Max:  {ndvi.max():.3f}

Vegetation Health Interpretation:
• NDVI > 0.6:  Healthy vegetation ({np.sum(ndvi > 0.6) / ndvi.size * 100:.1f}%)
• 0.3-0.6:     Moderate vegetation ({np.sum((ndvi > 0.3) & (ndvi <= 0.6)) / ndvi.size * 100:.1f}%)
• 0.0-0.3:     Sparse vegetation ({np.sum((ndvi > 0.0) & (ndvi <= 0.3)) / ndvi.size * 100:.1f}%)
• < 0.0:       Water/bare soil ({np.sum(ndvi <= 0.0) / ndvi.size * 100:.1f}%)
"""

axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('AgriFlux Satellite Imagery Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('satellite_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: satellite_visualization.png")
print("\nOpening image...")
plt.show()
