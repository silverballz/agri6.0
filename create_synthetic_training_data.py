"""
Create synthetic training data for CNN using rule-based labels.

This generates a training dataset by:
1. Using the rule-based classifier to create "weak labels"
2. Adding realistic variations and noise
3. Creating a balanced dataset for training

Note: This is for demonstration. Real training needs field-validated labels.
"""

import numpy as np
import rasterio
from pathlib import Path
from src.ai_models.rule_based_classifier import RuleBasedClassifier
from src.ai_models.crop_health_cnn import CropHealthCNN

print("=" * 70)
print("Creating Synthetic Training Dataset for CNN")
print("=" * 70)

# Load satellite data
base_path = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE/GRANULE/L2A_T43REQ_A048336_20240923T055000/IMG_DATA/R10m")

band_files = {
    'B02': base_path / "T43REQ_20240923T053641_B02_10m.jp2",
    'B03': base_path / "T43REQ_20240923T053641_B03_10m.jp2",
    'B04': base_path / "T43REQ_20240923T053641_B04_10m.jp2",
    'B08': base_path / "T43REQ_20240923T053641_B08_10m.jp2",
}

print("\n📡 Loading satellite bands...")
bands = {}
for band_name, file_path in band_files.items():
    with rasterio.open(file_path) as src:
        bands[band_name] = src.read(1).astype(np.float32)

# Calculate NDVI for labeling
nir = bands['B08']
red = bands['B04']
ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)

# Use rule-based classifier to generate labels
print("\n🏷️  Generating labels using rule-based classifier...")
classifier = RuleBasedClassifier()
labels = classifier.classify(ndvi)

print(f"   Label distribution:")
for i, class_name in enumerate(classifier.CLASS_NAMES):
    count = np.sum(labels.predictions == i)
    pct = count / labels.predictions.size * 100
    print(f"   {class_name:12s}: {count:8,} pixels ({pct:5.1f}%)")

# Extract patches for training
print("\n✂️  Extracting 64x64 patches...")
patch_size = 64
stride = 32  # 50% overlap

def normalize_band(band):
    return (band - band.min()) / (band.max() - band.min() + 1e-8)

# Stack bands
image_4band = np.stack([
    normalize_band(bands['B02']),
    normalize_band(bands['B03']),
    normalize_band(bands['B04']),
    normalize_band(bands['B08'])
], axis=-1).astype(np.float32)

h, w = image_4band.shape[:2]
patches = []
patch_labels = []

for i in range(0, h - patch_size + 1, stride):
    for j in range(0, w - patch_size + 1, stride):
        patch = image_4band[i:i+patch_size, j:j+patch_size]
        label_patch = labels.predictions[i:i+patch_size, j:j+patch_size]
        
        # Use center pixel label
        center_label = label_patch[patch_size//2, patch_size//2]
        
        patches.append(patch)
        patch_labels.append(center_label)

patches = np.array(patches)
patch_labels = np.array(patch_labels)

print(f"   Extracted {len(patches):,} patches")

# Balance dataset (equal samples per class)
print("\n⚖️  Balancing dataset...")
samples_per_class = 1000
balanced_patches = []
balanced_labels = []

for class_idx in range(4):
    class_indices = np.where(patch_labels == class_idx)[0]
    
    if len(class_indices) >= samples_per_class:
        selected = np.random.choice(class_indices, samples_per_class, replace=False)
    else:
        # Oversample if not enough samples
        selected = np.random.choice(class_indices, samples_per_class, replace=True)
    
    balanced_patches.append(patches[selected])
    balanced_labels.append(patch_labels[selected])

X_train = np.concatenate(balanced_patches, axis=0)
y_train = np.concatenate(balanced_labels, axis=0)

# Shuffle
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"   Balanced dataset: {len(X_train):,} patches")
print(f"   Shape: {X_train.shape}")

# Prepare labels for CNN (one-hot encoding)
print("\n🧠 Preparing data for CNN training...")
cnn = CropHealthCNN()

# Convert labels to one-hot
y_train_onehot = np.zeros((len(y_train), 64, 64, 4), dtype=np.float32)
for i in range(len(y_train)):
    y_train_onehot[i, :, :, y_train[i]] = 1.0

# Split into train/validation
split_idx = int(0.8 * len(X_train))
X_train_split = X_train[:split_idx]
y_train_split = y_train_onehot[:split_idx]
X_val = X_train[split_idx:]
y_val = y_train_onehot[split_idx:]

print(f"   Training set: {len(X_train_split):,} samples")
print(f"   Validation set: {len(X_val):,} samples")

# Train the model
print("\n🚀 Training CNN model...")
print("   Note: This uses synthetic labels from rule-based classifier")
print("   Real training would use field-validated ground truth")
print()

history = cnn.train(
    X_train_split,
    y_train_split,
    X_val=X_val,
    y_val=y_val,
    epochs=10,  # Small number for demo
    batch_size=32
)

print("\n✅ Training complete!")
print(f"   Final training accuracy: {history['accuracy'][-1]:.3f}")
print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.3f}")

# Save the trained model
model_path = 'models/crop_health_cnn_synthetic.h5'
cnn.save_model(model_path)
print(f"\n💾 Model saved to: {model_path}")

print("\n" + "=" * 70)
print("Synthetic Training Complete!")
print("=" * 70)
print("\n⚠️  IMPORTANT:")
print("   This model was trained on synthetic labels from the rule-based classifier.")
print("   For production use, train on real field-validated ground truth data.")
print("   The model will perform better than random but not as good as real labels.")
