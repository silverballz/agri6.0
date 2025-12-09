#!/usr/bin/env python3
"""
Generate Training Data for AI Models

This script generates comprehensive training data for both CNN and LSTM models:
1. Extracts 5000+ labeled patches from synthetic time series imagery
2. Uses rule-based classifier to generate weak labels
3. Creates 30-step time series sequences for LSTM training
4. Saves training data to data/training/

Usage:
    python scripts/generate_training_data.py [--num-patches 5000] [--sequence-length 30]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_models.rule_based_classifier import RuleBasedClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_synthetic_time_series(data_dir: Path) -> List[Dict]:
    """
    Load all synthetic time series data.
    
    Args:
        data_dir: Directory containing synthetic data
        
    Returns:
        List of dictionaries with date, bands, and indices
    """
    logger.info("Loading synthetic time series data...")
    
    # Find all synthetic date directories
    date_dirs = sorted([d for d in data_dir.glob('43REQ_*') if d.is_dir()])
    
    if not date_dirs:
        raise FileNotFoundError(f"No synthetic data found in {data_dir}")
    
    time_series = []
    
    for date_dir in date_dirs:
        # Load metadata
        metadata_file = date_dir / 'metadata.json'
        if not metadata_file.exists():
            logger.warning(f"Skipping {date_dir.name} - no metadata")
            continue
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load bands
        bands = {}
        for band_name in ['B02', 'B03', 'B04', 'B08']:
            band_file = date_dir / f"{band_name}.npy"
            if band_file.exists():
                bands[band_name] = np.load(band_file)
        
        # Load indices
        indices = {}
        for index_name in ['NDVI', 'SAVI', 'EVI', 'NDWI']:
            index_file = date_dir / f"{index_name}.npy"
            if index_file.exists():
                indices[index_name] = np.load(index_file)
        
        if bands and indices:
            time_series.append({
                'date': metadata['acquisition_date'],
                'date_dir': date_dir.name,
                'bands': bands,
                'indices': indices,
                'metadata': metadata
            })
            logger.info(f"  Loaded {date_dir.name}: {metadata['acquisition_date']}")
    
    logger.info(f"Loaded {len(time_series)} dates")
    return time_series


def generate_cnn_training_patches(
    time_series: List[Dict],
    num_patches: int = 5000,
    patch_size: int = 64,
    stride: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training patches for CNN.
    
    Args:
        time_series: List of time series data
        num_patches: Target number of patches
        patch_size: Size of each patch
        stride: Stride for patch extraction
        
    Returns:
        Tuple of (X_train, y_train)
    """
    logger.info(f"Generating {num_patches} CNN training patches...")
    logger.info(f"  Patch size: {patch_size}x{patch_size}")
    logger.info(f"  Stride: {stride}")
    
    classifier = RuleBasedClassifier()
    
    all_patches = []
    all_labels = []
    
    # Extract patches from each date
    for ts_data in time_series:
        bands = ts_data['bands']
        indices = ts_data['indices']
        
        # Normalize bands
        def normalize(band):
            return (band - band.min()) / (band.max() - band.min() + 1e-8)
        
        # Stack bands into 4-channel image
        image_4band = np.stack([
            normalize(bands['B02']),
            normalize(bands['B03']),
            normalize(bands['B04']),
            normalize(bands['B08'])
        ], axis=-1).astype(np.float32)
        
        # Get NDVI for labeling
        ndvi = indices['NDVI']
        
        # Generate labels using rule-based classifier
        labels = classifier.classify(ndvi)
        
        # Extract patches
        h, w = image_4band.shape[:2]
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = image_4band[i:i+patch_size, j:j+patch_size]
                label_patch = labels.predictions[i:i+patch_size, j:j+patch_size]
                
                # Use center pixel label
                center_label = label_patch[patch_size//2, patch_size//2]
                
                all_patches.append(patch)
                all_labels.append(center_label)
                
                if len(all_patches) >= num_patches * 2:  # Extract more than needed
                    break
            
            if len(all_patches) >= num_patches * 2:
                break
        
        if len(all_patches) >= num_patches * 2:
            break
    
    logger.info(f"  Extracted {len(all_patches)} patches total")
    
    # Convert to arrays
    X = np.array(all_patches)
    y = np.array(all_labels)
    
    # Balance dataset (equal samples per class)
    logger.info("  Balancing dataset...")
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    logger.info(f"  Class distribution before balancing:")
    for cls, count in zip(unique_classes, class_counts):
        logger.info(f"    Class {cls} ({classifier.CLASS_NAMES[cls]}): {count} samples")
    
    # Find classes with samples
    available_classes = unique_classes
    samples_per_class = num_patches // len(available_classes)
    
    balanced_X = []
    balanced_y = []
    
    for class_idx in available_classes:
        class_indices = np.where(y == class_idx)[0]
        
        if len(class_indices) == 0:
            logger.warning(f"    Class {class_idx} has no samples, skipping")
            continue
        
        if len(class_indices) >= samples_per_class:
            selected = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            # Oversample if not enough samples
            selected = np.random.choice(class_indices, samples_per_class, replace=True)
        
        balanced_X.append(X[selected])
        balanced_y.append(y[selected])
        
        logger.info(f"    Class {class_idx} ({classifier.CLASS_NAMES[class_idx]}): {len(selected)} samples")
    
    X_train = np.concatenate(balanced_X, axis=0)
    y_train = np.concatenate(balanced_y, axis=0)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    logger.info(f"  Final dataset: {len(X_train)} patches")
    logger.info(f"  Shape: {X_train.shape}")
    
    return X_train, y_train


def generate_lstm_time_series(
    time_series: List[Dict],
    sequence_length: int = 30,
    num_sequences: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time series sequences for LSTM training.
    
    Args:
        time_series: List of time series data
        sequence_length: Length of each sequence
        num_sequences: Number of sequences to generate
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    logger.info(f"Generating {num_sequences} LSTM time series sequences...")
    logger.info(f"  Sequence length: {sequence_length}")
    
    if len(time_series) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} dates, got {len(time_series)}")
    
    # Sort by date
    time_series = sorted(time_series, key=lambda x: x['date'])
    
    # Sample random pixels from the imagery
    first_ndvi = time_series[0]['indices']['NDVI']
    h, w = first_ndvi.shape
    
    sequences = []
    targets = []
    
    for _ in range(num_sequences):
        # Random pixel location
        i = np.random.randint(0, h)
        j = np.random.randint(0, w)
        
        # Extract time series for this pixel
        pixel_series = []
        
        for ts_data in time_series:
            indices = ts_data['indices']
            
            # Create feature vector: [NDVI, SAVI, EVI, NDWI]
            features = np.array([
                indices['NDVI'][i, j],
                indices['SAVI'][i, j],
                indices['EVI'][i, j],
                indices['NDWI'][i, j]
            ])
            
            pixel_series.append(features)
        
        pixel_series = np.array(pixel_series)
        
        # Create sequences (sliding window)
        for start_idx in range(len(pixel_series) - sequence_length):
            seq = pixel_series[start_idx:start_idx + sequence_length]
            target = pixel_series[start_idx + sequence_length, 0]  # Predict next NDVI
            
            sequences.append(seq)
            targets.append(target)
            
            if len(sequences) >= num_sequences:
                break
        
        if len(sequences) >= num_sequences:
            break
    
    X_sequences = np.array(sequences[:num_sequences])
    y_sequences = np.array(targets[:num_sequences])
    
    logger.info(f"  Generated {len(X_sequences)} sequences")
    logger.info(f"  X shape: {X_sequences.shape}")
    logger.info(f"  y shape: {y_sequences.shape}")
    
    return X_sequences, y_sequences


def save_training_data(
    X_cnn: np.ndarray,
    y_cnn: np.ndarray,
    X_lstm: np.ndarray,
    y_lstm: np.ndarray,
    output_dir: Path
):
    """
    Save training data to disk.
    
    Args:
        X_cnn: CNN training patches
        y_cnn: CNN labels
        X_lstm: LSTM sequences
        y_lstm: LSTM targets
        output_dir: Output directory
    """
    logger.info(f"Saving training data to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CNN data
    np.save(output_dir / 'cnn_X_train.npy', X_cnn)
    np.save(output_dir / 'cnn_y_train.npy', y_cnn)
    logger.info(f"  Saved CNN data: {X_cnn.shape}, {y_cnn.shape}")
    
    # Save LSTM data
    np.save(output_dir / 'lstm_X_train.npy', X_lstm)
    np.save(output_dir / 'lstm_y_train.npy', y_lstm)
    logger.info(f"  Saved LSTM data: {X_lstm.shape}, {y_lstm.shape}")
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'cnn': {
            'num_samples': int(len(X_cnn)),
            'patch_size': int(X_cnn.shape[1]),
            'num_channels': int(X_cnn.shape[3]),
            'num_classes': 4,
            'class_names': ['healthy', 'moderate', 'stressed', 'critical']
        },
        'lstm': {
            'num_sequences': int(len(X_lstm)),
            'sequence_length': int(X_lstm.shape[1]),
            'num_features': int(X_lstm.shape[2])
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("  Saved metadata.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate training data for AI models'
    )
    parser.add_argument(
        '--num-patches',
        type=int,
        default=5000,
        help='Number of CNN training patches (default: 5000)'
    )
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=1000,
        help='Number of LSTM sequences (default: 1000)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=10,
        help='Length of LSTM sequences (default: 10)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=64,
        help='Size of CNN patches (default: 64)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/processed'),
        help='Directory containing synthetic time series data'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training'),
        help='Output directory for training data'
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("="*70)
    logger.info("Generating Training Data for AI Models")
    logger.info("="*70)
    
    try:
        # Load synthetic time series
        time_series = load_synthetic_time_series(args.data_dir)
        
        if not time_series:
            logger.error("No time series data found")
            sys.exit(1)
        
        # Generate CNN training patches
        logger.info("\n" + "="*70)
        logger.info("CNN Training Data Generation")
        logger.info("="*70)
        X_cnn, y_cnn = generate_cnn_training_patches(
            time_series,
            num_patches=args.num_patches,
            patch_size=args.patch_size
        )
        
        # Generate LSTM time series
        logger.info("\n" + "="*70)
        logger.info("LSTM Training Data Generation")
        logger.info("="*70)
        X_lstm, y_lstm = generate_lstm_time_series(
            time_series,
            sequence_length=args.sequence_length,
            num_sequences=args.num_sequences
        )
        
        # Save training data
        logger.info("\n" + "="*70)
        logger.info("Saving Training Data")
        logger.info("="*70)
        save_training_data(X_cnn, y_cnn, X_lstm, y_lstm, args.output_dir)
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING DATA GENERATION SUMMARY")
        print("="*70)
        print(f"CNN Training Data:")
        print(f"  Patches: {len(X_cnn):,}")
        print(f"  Shape: {X_cnn.shape}")
        print(f"  Classes: 4 (healthy, moderate, stressed, critical)")
        print(f"\nLSTM Training Data:")
        print(f"  Sequences: {len(X_lstm):,}")
        print(f"  Shape: {X_lstm.shape}")
        print(f"  Sequence length: {args.sequence_length}")
        print(f"\nOutput directory: {args.output_dir}")
        print("="*70)
        
        logger.info("\n✅ Training data generation complete!")
        logger.info("Ready to train AI models.")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
