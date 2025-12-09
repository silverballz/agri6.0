"""
Complete training pipeline: CNN + LSTM + Synthetic Sensors

This script demonstrates the full AgriFlux system:
1. Load satellite data
2. Generate synthetic sensor data
3. Train CNN on synthetic labels
4. Simulate time-series for LSTM
5. Integrate everything
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from datetime import datetime, timedelta
import json

print("=" * 80)
print("AgriFlux Complete Training Pipeline")
print("=" * 80)

# Import models
from src.ai_models.crop_health_cnn import CropHealthCNN
from src.ai_models.vegetation_trend_lstm import VegetationTrendLSTM
from src.ai_models.rule_based_classifier import RuleBasedClassifier

# Load satellite data (smaller crop for memory efficiency)
base_path = Path("S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE/GRANULE/L2A_T43REQ_A048336_20240923T055000/IMG_DATA/R10m")

print("\n📡 Step 1: Loading Satellite Data...")
bands = {}
for band_name in ['B02', 'B03', 'B04', 'B08']:
    file_path = base_path / f"T43REQ_20240923T053641_{band_name}_10m.jp2"
    with rasterio.open(file_path) as src:
        # Load smaller region for efficiency
        bands[band_name] = src.read(1, window=((2000, 3000), (2000, 3000))).astype(np.float32)

print(f"   Loaded {len(bands)} bands, shape: {bands['B02'].shape}")

# Calculate NDVI
nir = bands['B08']
red = bands['B04']
ndvi = (nir - red) / (nir + red + 1e-8)
ndvi = np.clip(ndvi, -1, 1)

print(f"   NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}], mean: {ndvi.mean():.3f}")

# Step 2: Generate Synthetic Sensor Data
print("\n🌡️  Step 2: Generating Synthetic Sensor Data...")

# Soil moisture (correlated with NDVI)
soil_moisture = 10 + (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-8) * 30
soil_moisture += np.random.normal(0, 2, soil_moisture.shape)
soil_moisture = np.clip(soil_moisture, 0, 50)

# Temperature (seasonal + daily variation)
base_temp = 28  # September in Punjab
temperature = base_temp + np.random.normal(0, 3, ndvi.shape)
temperature = np.clip(temperature, 15, 40)

# Humidity (inversely correlated with temperature)
humidity = 80 - (temperature - 20) * 1.5 + soil_moisture * 0.3
humidity += np.random.normal(0, 5, humidity.shape)
humidity = np.clip(humidity, 30, 95)

print(f"   Soil Moisture: {soil_moisture.mean():.1f}% ± {soil_moisture.std():.1f}%")
print(f"   Temperature: {temperature.mean():.1f}°C ± {temperature.std():.1f}°C")
print(f"   Humidity: {humidity.mean():.1f}% ± {humidity.std():.1f}%")

# Step 3: Train CNN
print("\n🧠 Step 3: Training CNN Model...")

# Generate labels using rule-based classifier
classifier = RuleBasedClassifier()
labels = classifier.classify(ndvi)

print(f"   Label distribution:")
for i, name in enumerate(classifier.CLASS_NAMES):
    count = np.sum(labels.predictions == i)
    pct = count / labels.predictions.size * 100
    print(f"     {name:12s}: {pct:5.1f}%")

# Prepare CNN training data (small patches)
patch_size = 64
stride = 128  # Less overlap for speed
patches = []
patch_labels = []

def normalize(band):
    return (band - band.min()) / (band.max() - band.min() + 1e-8)

image_4band = np.stack([normalize(bands[b]) for b in ['B02', 'B03', 'B04', 'B08']], axis=-1).astype(np.float32)

h, w = image_4band.shape[:2]
for i in range(0, h - patch_size + 1, stride):
    for j in range(0, w - patch_size + 1, stride):
        patches.append(image_4band[i:i+patch_size, j:j+patch_size])
        patch_labels.append(labels.predictions[i+patch_size//2, j+patch_size//2])

patches = np.array(patches[:200])  # Limit for speed
patch_labels = np.array(patch_labels[:200])

print(f"   Extracted {len(patches)} patches")

# Train CNN (small epochs for demo)
cnn = CropHealthCNN()

# Convert labels to one-hot encoding manually
y_train_onehot = np.zeros((len(patches), 64, 64, 4), dtype=np.float32)
for i in range(len(patches)):
    y_train_onehot[i, :, :, patch_labels[i]] = 1.0

print(f"   Training CNN (5 epochs for demo)...")
history = cnn.train(patches, y_train_onehot, epochs=5, batch_size=16)

print(f"   ✓ Training complete! Final accuracy: {history['accuracy'][-1]:.3f}")

# Save model
cnn.save_model('models/cnn_trained_demo.h5')
print(f"   ✓ Model saved")

# Step 4: Simulate Time-Series for LSTM
print("\n📈 Step 4: Simulating Time-Series Data for LSTM...")

# Create synthetic time-series (simulate 30 days)
dates = [datetime(2024, 9, 23) - timedelta(days=i) for i in range(30, 0, -1)]
dates_str = [d.strftime("%Y-%m-%d") for d in dates]

# Sample a few pixels and create time series
n_samples = 100
sample_indices = np.random.choice(ndvi.size, n_samples, replace=False)

time_series_data = []
for idx in sample_indices:
    i, j = np.unravel_index(idx, ndvi.shape)
    
    # Create realistic time series with trend
    base_ndvi = ndvi[i, j]
    trend = np.linspace(base_ndvi - 0.1, base_ndvi, 30)
    noise = np.random.normal(0, 0.02, 30)
    ndvi_series = np.clip(trend + noise, 0, 1)
    
    # Corresponding sensor data
    temp_series = temperature[i, j] + np.random.normal(0, 2, 30)
    humid_series = humidity[i, j] + np.random.normal(0, 3, 30)
    soil_series = soil_moisture[i, j] + np.random.normal(0, 1, 30)
    
    for t in range(30):
        time_series_data.append({
            'date': dates_str[t],
            'pixel_id': idx,
            'ndvi': ndvi_series[t],
            'temperature': temp_series[t],
            'humidity': humid_series[t],
            'soil_moisture': soil_series[t]
        })

df = pd.DataFrame(time_series_data)
print(f"   Created time-series with {len(df)} data points")
print(f"   Covering {n_samples} pixels over 30 days")

# Train LSTM
print("\n🔮 Step 5: Training LSTM Model...")
lstm = VegetationTrendLSTM(sequence_length=10)

# Prepare sequences for one pixel
pixel_data = df[df['pixel_id'] == sample_indices[0]].sort_values('date')
pixel_df = pixel_data[['ndvi', 'temperature', 'humidity', 'soil_moisture']].reset_index(drop=True)

X_lstm, y_lstm = lstm.prepare_sequences(pixel_df, target_column='ndvi')

print(f"   Prepared {len(X_lstm)} sequences")
print(f"   Training LSTM (10 epochs for demo)...")

history_lstm = lstm.train(X_lstm, y_lstm, epochs=10, batch_size=8)

print(f"   ✓ Training complete! Final MAE: {history_lstm['mae'][-1]:.4f}")

# Save model
lstm.save_model('models/lstm_trained_demo.h5')
print(f"   ✓ Model saved")

# Step 6: Test Predictions
print("\n🎯 Step 6: Testing Trained Models...")

# Test CNN
test_patch = patches[0:1]
pred, conf = cnn.predict_with_confidence(test_patch)
print(f"   CNN Prediction: {classifier.CLASS_NAMES[pred[0, 32, 32]]} (confidence: {conf[0, 32, 32]:.3f})")

# Test LSTM
test_seq = X_lstm[0:1]
pred_lstm, _, trend_dir, trend_str = lstm.predict_trend(test_seq, return_confidence=False)
print(f"   LSTM Prediction: {pred_lstm[0]:.3f}, Trend: {trend_dir} (strength: {trend_str:.3f})")

# Save summary
summary = {
    'training_date': datetime.now().isoformat(),
    'cnn': {
        'patches_trained': len(patches),
        'final_accuracy': float(history['accuracy'][-1]),
        'model_path': 'models/cnn_trained_demo.h5'
    },
    'lstm': {
        'sequences_trained': len(X_lstm),
        'final_mae': float(history_lstm['mae'][-1]),
        'model_path': 'models/lstm_trained_demo.h5'
    },
    'sensor_data': {
        'soil_moisture_mean': float(soil_moisture.mean()),
        'temperature_mean': float(temperature.mean()),
        'humidity_mean': float(humidity.mean())
    }
}

with open('training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("✅ Complete Training Pipeline Finished!")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"   CNN: Trained on {len(patches)} patches, accuracy: {history['accuracy'][-1]:.3f}")
print(f"   LSTM: Trained on {len(X_lstm)} sequences, MAE: {history_lstm['mae'][-1]:.4f}")
print(f"   Synthetic sensors: Soil moisture, temperature, humidity generated")
print(f"\n💾 Models saved:")
print(f"   - models/cnn_trained_demo.h5")
print(f"   - models/lstm_trained_demo.h5")
print(f"   - training_summary.json")
print(f"\n🎉 System is now ready for predictions!")
