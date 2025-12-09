"""
Run AI models on real Sentinel-2A satellite imagery.

This demonstrates the CNN model and rule-based classifier making
crop health predictions on actual satellite data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from pathlib import Path

# Import our AI models
from src.ai_models.crop_health_cnn import CropHealthCNN
from src.ai_models.rule_based_classifier import RuleBasedClassifier

print("=" * 70)
print("AgriFlux AI Crop Health Prediction Demo")
print("=" * 70)

# Load satellite bands
base_path = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE/GRANULE/L2A_T43REQ_A048336_20240923T055000/IMG_DATA/R10m")

band_files = {
    'B02': base_path / "T43REQ_20240923T053641_B02_10m.jp2",  # Blue
    'B03': base_path / "T43REQ_20240923T053641_B03_10m.jp2",  # Green
    'B04': base_path / "T43REQ_20240923T053641_B04_10m.jp2",  # Red
    'B08': base_path / "T43REQ_20240923T053641_B08_10m.jp2",  # NIR
}

print("\n📡 Loading Sentinel-2A satellite data...")
bands = {}
for band_name, file_path in band_files.items():
    with rasterio.open(file_path) as src:
        bands[band_name] = src.read(1).astype(np.float32)

# Get a random crop (smaller for faster processing)
crop_size = 512
h, w = bands['B02'].shape
start_y = np.random.randint(0, h - crop_size)
start_x = np.random.randint(0, w - crop_size)

print(f"   Cropping region: [{start_y}:{start_y+crop_size}, {start_x}:{start_x+crop_size}]")
print(f"   Area: ~{crop_size*10/1000:.1f} × {crop_size*10/1000:.1f} km")

# Crop all bands
for band_name in bands:
    bands[band_name] = bands[band_name][start_y:start_y+crop_size, start_x:start_x+crop_size]

# Calculate NDVI
nir = bands['B08']
red = bands['B04']
ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)

print(f"\n📊 NDVI Statistics:")
print(f"   Mean: {ndvi.mean():.3f}")
print(f"   Range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")

# Prepare data for CNN (normalize to 0-1 range)
def normalize_band(band):
    return (band - band.min()) / (band.max() - band.min() + 1e-8)

# Stack bands for CNN input
image_4band = np.stack([
    normalize_band(bands['B02']),
    normalize_band(bands['B03']),
    normalize_band(bands['B04']),
    normalize_band(bands['B08'])
], axis=-1).astype(np.float32)

print(f"\n🤖 Initializing AI Models...")

# Initialize CNN
cnn = CropHealthCNN()
print(f"   ✓ CNN Model: {cnn.model.count_params() if cnn.model else 0:,} parameters")

# Initialize Rule-Based Classifier
rule_classifier = RuleBasedClassifier()
print(f"   ✓ Rule-Based Classifier: Ready")

# Run Rule-Based Classification (always works)
print(f"\n🌾 Running Rule-Based Classification...")
rule_result = rule_classifier.classify(ndvi)
print(f"   ✓ Classification complete!")

# Get statistics
stats = rule_classifier.get_class_statistics(rule_result)
print(f"\n📈 Crop Health Distribution:")
for class_name, class_stats in stats.items():
    print(f"   {class_name:12s}: {class_stats['percentage']:5.1f}% ({class_stats['count']:,} pixels)")

# Try CNN prediction (will use untrained model)
print(f"\n🧠 Running CNN Model...")
print(f"   Note: Model is untrained, so predictions are random")
print(f"   (In production, this would use a trained model)")

# Extract 64x64 patches for CNN
patch_size = 64
stride = 64  # Non-overlapping patches
patches = []
patch_positions = []

for i in range(0, crop_size - patch_size + 1, stride):
    for j in range(0, crop_size - patch_size + 1, stride):
        patch = image_4band[i:i+patch_size, j:j+patch_size]
        patches.append(patch)
        patch_positions.append((i, j))

patches = np.array(patches)
print(f"   Extracted {len(patches)} patches of size {patch_size}×{patch_size}")

# Mark CNN as trained for demo purposes
cnn.is_trained = True

# Run CNN predictions
print(f"   Running inference...")
cnn_predictions, cnn_confidence = cnn.predict_batch(patches, batch_size=16)

# Reconstruct full image from patches
cnn_prediction_map = np.zeros((crop_size, crop_size), dtype=np.int32)
cnn_confidence_map = np.zeros((crop_size, crop_size), dtype=np.float32)

for idx, (i, j) in enumerate(patch_positions):
    pred = cnn_predictions[idx]
    conf = cnn_confidence[idx]
    
    # Take center pixel of each patch
    center_i = i + patch_size // 2
    center_j = j + patch_size // 2
    
    if center_i < crop_size and center_j < crop_size:
        cnn_prediction_map[center_i, center_j] = pred[patch_size//2, patch_size//2]
        cnn_confidence_map[center_i, center_j] = conf[patch_size//2, patch_size//2]

print(f"   ✓ CNN inference complete!")
print(f"   Average confidence: {cnn_confidence.mean():.3f}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Color maps for health classes
health_colors = ['#228B22', '#90EE90', '#FFD700', '#FF4500']  # Green, Light Green, Gold, Red
health_cmap = ListedColormap(health_colors)
class_names = ['Healthy', 'Moderate', 'Stressed', 'Critical']

# 1. True Color RGB
rgb = np.stack([
    normalize_band(bands['B04']),
    normalize_band(bands['B03']),
    normalize_band(bands['B02'])
], axis=-1)

axes[0, 0].imshow(rgb)
axes[0, 0].set_title('True Color Satellite Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# 2. NDVI
from matplotlib.colors import LinearSegmentedColormap
ndvi_cmap = LinearSegmentedColormap.from_list('ndvi', [
    '#8B4513', '#FFFF00', '#90EE90', '#228B22', '#006400'
])
im1 = axes[0, 1].imshow(ndvi, cmap=ndvi_cmap, vmin=-0.2, vmax=0.9)
axes[0, 1].set_title('NDVI (Vegetation Index)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

# 3. Rule-Based Classification
im2 = axes[0, 2].imshow(rule_result.predictions, cmap=health_cmap, vmin=0, vmax=3)
axes[0, 2].set_title('Rule-Based Classification\n(NDVI Thresholds)', fontsize=12, fontweight='bold')
axes[0, 2].axis('off')
cbar2 = plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
cbar2.ax.set_yticklabels(class_names)

# 4. Rule-Based Confidence
im3 = axes[1, 0].imshow(rule_result.confidence_scores, cmap='RdYlGn', vmin=0, vmax=1)
axes[1, 0].set_title('Rule-Based Confidence', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Confidence')

# 5. CNN Classification (sparse due to patch-based approach)
im4 = axes[1, 1].imshow(cnn_prediction_map, cmap=health_cmap, vmin=0, vmax=3)
axes[1, 1].set_title('CNN Classification\n(Untrained - Random)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
cbar4.ax.set_yticklabels(class_names)

# 6. Statistics
axes[1, 2].axis('off')
info_text = f"""
🌾 CROP HEALTH ANALYSIS

Date: September 23, 2024
Location: Ludhiana, Punjab
Area: {crop_size*10/1000:.1f} × {crop_size*10/1000:.1f} km

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE-BASED CLASSIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

for class_name, class_stats in stats.items():
    bar = '█' * int(class_stats['percentage'] / 2)
    info_text += f"\n{class_name:10s} {class_stats['percentage']:5.1f}% {bar}"

info_text += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NDVI STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mean:  {ndvi.mean():.3f}
Std:   {ndvi.std():.3f}
Range: [{ndvi.min():.3f}, {ndvi.max():.3f}]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Rule-Based: Active
⚠ CNN: Untrained (random)
⚠ LSTM: Not shown (needs time series)

Note: CNN shows random predictions
because it's untrained. In production,
it would be trained on labeled data.
"""

axes[1, 2].text(0.05, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('AgriFlux AI Crop Health Predictions', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('ai_predictions.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: ai_predictions.png")
print("\nOpening image...")
plt.show()

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
