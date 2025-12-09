#!/usr/bin/env python3
"""
Prepare Training Data from Real Satellite Imagery

This script prepares training datasets from real Sentinel-2 imagery for CNN model training:
1. Finds only real (non-synthetic) imagery directories
2. Extracts 64x64 patches from real imagery
3. Generates labels using rule-based classifier on NDVI
4. Balances dataset across health classes
5. Splits into train/validation (80/20)
6. Saves prepared data with real data metadata

Usage:
    python scripts/prepare_real_training_data.py --samples-per-class 2000
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import json

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_models.rule_based_classifier import RuleBasedClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/real_training_data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealDatasetPreparator:
    """
    Prepare training datasets from real satellite imagery.
    
    This class ensures that only real (non-synthetic) imagery is used for training,
    as specified in the requirements.
    """
    
    def __init__(self, processed_dir: Path, output_dir: Path):
        """
        Initialize dataset preparator.
        
        Args:
            processed_dir: Directory containing processed imagery
            output_dir: Directory for output training data
        """
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.classifier = RuleBasedClassifier()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RealDatasetPreparator initialized")
        logger.info(f"Processed imagery directory: {self.processed_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _find_real_imagery_dirs(self) -> List[Path]:
        """
        Find all directories containing real (non-synthetic) imagery.
        
        Only includes directories where metadata.json has synthetic=false.
        
        Returns:
            List of paths to real imagery directories, sorted by date
        """
        logger.info("Searching for real imagery directories...")
        
        real_dirs = []
        
        # Iterate through all subdirectories
        for img_dir in self.processed_dir.iterdir():
            if not img_dir.is_dir():
                continue
            
            # Check for metadata file
            metadata_file = img_dir / 'metadata.json'
            if not metadata_file.exists():
                logger.debug(f"  Skipping {img_dir.name} - no metadata.json")
                continue
            
            # Load metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"  Skipping {img_dir.name} - failed to load metadata: {e}")
                continue
            
            # Check if synthetic flag is false (real data)
            is_synthetic = metadata.get('synthetic', True)
            
            if not is_synthetic:
                real_dirs.append(img_dir)
                logger.info(f"  ✓ Found real imagery: {img_dir.name}")
            else:
                logger.debug(f"  Skipping {img_dir.name} - synthetic data")
        
        # Sort by directory name (which includes date)
        real_dirs = sorted(real_dirs)
        
        logger.info(f"Found {len(real_dirs)} real imagery directories")
        
        if len(real_dirs) == 0:
            logger.warning("No real imagery found! Make sure to run download_real_satellite_data.py first")
        
        return real_dirs
    
    def _extract_patches_from_imagery(
        self,
        img_dir: Path,
        patch_size: int,
        stride: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract patches from a single imagery directory.
        
        Args:
            img_dir: Path to imagery directory
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
            
        Returns:
            Tuple of (patches, labels) arrays
        """
        logger.debug(f"  Extracting patches from {img_dir.name}...")
        
        # Load bands
        bands = {}
        for band_name in ['B02', 'B03', 'B04', 'B08']:
            band_file = img_dir / f"{band_name}.npy"
            if not band_file.exists():
                raise FileNotFoundError(f"Band file not found: {band_file}")
            bands[band_name] = np.load(band_file)
        
        # Load NDVI for labeling
        ndvi_file = img_dir / 'NDVI.npy'
        if not ndvi_file.exists():
            raise FileNotFoundError(f"NDVI file not found: {ndvi_file}")
        ndvi = np.load(ndvi_file)
        
        # Normalize bands to [0, 1] range
        def normalize(band):
            # Handle edge cases
            band_min = np.nanmin(band)
            band_max = np.nanmax(band)
            if band_max - band_min < 1e-8:
                return np.zeros_like(band, dtype=np.float32)
            return ((band - band_min) / (band_max - band_min)).astype(np.float32)
        
        # Stack bands into 4-channel image [H, W, 4]
        image_4band = np.stack([
            normalize(bands['B02']),
            normalize(bands['B03']),
            normalize(bands['B04']),
            normalize(bands['B08'])
        ], axis=-1)
        
        # Generate labels using rule-based classifier
        labels_result = self.classifier.classify(ndvi)
        labels = labels_result.predictions
        
        # Extract patches
        patches = []
        patch_labels = []
        
        h, w = image_4band.shape[:2]
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # Extract patch
                patch = image_4band[i:i+patch_size, j:j+patch_size]
                label_patch = labels[i:i+patch_size, j:j+patch_size]
                
                # Use center pixel label
                center_label = label_patch[patch_size//2, patch_size//2]
                
                # Skip patches with NaN values
                if np.any(np.isnan(patch)):
                    continue
                
                patches.append(patch)
                patch_labels.append(center_label)
        
        logger.debug(f"    Extracted {len(patches)} patches")
        
        return np.array(patches), np.array(patch_labels)
    
    def prepare_cnn_dataset(
        self,
        patch_size: int = 64,
        stride: int = 32,
        samples_per_class: int = 2000,
        train_split: float = 0.8
    ) -> Dict[str, np.ndarray]:
        """
        Prepare CNN training dataset from real imagery.
        
        Args:
            patch_size: Size of patches (default: 64x64)
            stride: Stride for patch extraction (default: 32)
            samples_per_class: Number of samples per class (default: 2000)
            train_split: Fraction for training set (default: 0.8)
            
        Returns:
            Dictionary with X_train, y_train, X_val, y_val arrays
        """
        logger.info("="*70)
        logger.info("Preparing CNN Training Dataset from Real Imagery")
        logger.info("="*70)
        logger.info(f"Parameters:")
        logger.info(f"  Patch size: {patch_size}x{patch_size}")
        logger.info(f"  Stride: {stride}")
        logger.info(f"  Samples per class: {samples_per_class}")
        logger.info(f"  Train/Val split: {train_split:.0%}/{1-train_split:.0%}")
        
        # Find all real imagery directories
        imagery_dirs = self._find_real_imagery_dirs()
        
        if not imagery_dirs:
            raise ValueError(
                "No real imagery found! Please run download_real_satellite_data.py first."
            )
        
        # Extract patches from all real imagery
        logger.info(f"\nExtracting patches from {len(imagery_dirs)} imagery dates...")
        
        all_patches = []
        all_labels = []
        
        for i, img_dir in enumerate(imagery_dirs, 1):
            logger.info(f"Processing {i}/{len(imagery_dirs)}: {img_dir.name}")
            
            try:
                patches, labels = self._extract_patches_from_imagery(
                    img_dir,
                    patch_size,
                    stride
                )
                all_patches.append(patches)
                all_labels.append(labels)
                
            except Exception as e:
                logger.error(f"  Failed to extract patches: {e}")
                continue
        
        if not all_patches:
            raise ValueError("Failed to extract any patches from real imagery")
        
        # Concatenate all patches
        logger.info("\nCombining patches from all dates...")
        X = np.concatenate(all_patches, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        logger.info(f"  Total patches extracted: {len(X):,}")
        logger.info(f"  Patch shape: {X.shape}")
        
        # Balance dataset
        logger.info("\nBalancing dataset across health classes...")
        X_balanced, y_balanced = self._balance_dataset(X, y, samples_per_class)
        
        # Split into train/validation
        logger.info("\nSplitting into train/validation sets...")
        dataset = self._train_val_split(X_balanced, y_balanced, train_split)
        
        # Save prepared data
        logger.info("\nSaving prepared training data...")
        self._save_training_data(dataset, 'cnn')
        
        logger.info("\n" + "="*70)
        logger.info("CNN Dataset Preparation Complete!")
        logger.info("="*70)
        logger.info(f"Training samples: {len(dataset['X_train']):,}")
        logger.info(f"Validation samples: {len(dataset['X_val']):,}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*70)
        
        return dataset
    
    def _balance_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        samples_per_class: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset to have equal samples per class.
        
        Args:
            X: Input patches
            y: Labels
            samples_per_class: Target number of samples per class
            
        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        # Check class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        logger.info("  Class distribution before balancing:")
        for cls, count in zip(unique_classes, class_counts):
            class_name = self.classifier.CLASS_NAMES[cls]
            logger.info(f"    Class {cls} ({class_name}): {count:,} samples")
        
        # Balance each class
        balanced_X = []
        balanced_y = []
        
        for class_idx in range(len(self.classifier.CLASS_NAMES)):
            class_indices = np.where(y == class_idx)[0]
            
            if len(class_indices) == 0:
                logger.warning(f"    Class {class_idx} ({self.classifier.CLASS_NAMES[class_idx]}): "
                             f"No samples found, skipping")
                continue
            
            if len(class_indices) >= samples_per_class:
                # Randomly sample without replacement
                selected = np.random.choice(class_indices, samples_per_class, replace=False)
            else:
                # Oversample with replacement if not enough samples
                logger.warning(f"    Class {class_idx} ({self.classifier.CLASS_NAMES[class_idx]}): "
                             f"Only {len(class_indices)} samples, oversampling to {samples_per_class}")
                selected = np.random.choice(class_indices, samples_per_class, replace=True)
            
            balanced_X.append(X[selected])
            balanced_y.append(y[selected])
            
            logger.info(f"    Class {class_idx} ({self.classifier.CLASS_NAMES[class_idx]}): "
                       f"{len(selected):,} samples selected")
        
        # Concatenate all classes
        X_balanced = np.concatenate(balanced_X, axis=0)
        y_balanced = np.concatenate(balanced_y, axis=0)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        logger.info(f"  Balanced dataset: {len(X_balanced):,} samples")
        
        return X_balanced, y_balanced
    
    def _train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train and validation sets.
        
        Args:
            X: Input data
            y: Labels
            train_split: Fraction for training set
            
        Returns:
            Dictionary with X_train, y_train, X_val, y_val
        """
        # Calculate split index
        n_train = int(len(X) * train_split)
        
        # Split data
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:]
        y_val = y[n_train:]
        
        logger.info(f"  Training set: {len(X_train):,} samples ({train_split:.0%})")
        logger.info(f"  Validation set: {len(X_val):,} samples ({1-train_split:.0%})")
        
        # Check class distribution in splits
        logger.info("  Training set class distribution:")
        for class_idx in range(len(self.classifier.CLASS_NAMES)):
            count = np.sum(y_train == class_idx)
            pct = count / len(y_train) * 100
            logger.info(f"    Class {class_idx} ({self.classifier.CLASS_NAMES[class_idx]}): "
                       f"{count:,} ({pct:.1f}%)")
        
        logger.info("  Validation set class distribution:")
        for class_idx in range(len(self.classifier.CLASS_NAMES)):
            count = np.sum(y_val == class_idx)
            pct = count / len(y_val) * 100
            logger.info(f"    Class {class_idx} ({self.classifier.CLASS_NAMES[class_idx]}): "
                       f"{count:,} ({pct:.1f}%)")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }
    
    def _save_training_data(
        self,
        dataset: Dict[str, np.ndarray],
        dataset_type: str
    ):
        """
        Save prepared training data to disk with metadata.
        
        Args:
            dataset: Dictionary with train/val data
            dataset_type: Type of dataset ('cnn' or 'lstm')
        """
        # Save arrays
        for key, array in dataset.items():
            filename = f"{dataset_type}_{key}_real.npy"
            filepath = self.output_dir / filename
            np.save(filepath, array)
            logger.info(f"  Saved {filename}: shape={array.shape}")
        
        # Create metadata
        metadata = {
            'dataset_type': dataset_type,
            'data_source': 'real',
            'created_at': datetime.now().isoformat(),
            'num_train_samples': int(len(dataset['X_train'])),
            'num_val_samples': int(len(dataset['X_val'])),
            'num_classes': len(self.classifier.CLASS_NAMES),
            'class_names': self.classifier.CLASS_NAMES,
            'patch_size': int(dataset['X_train'].shape[1]),
            'num_channels': int(dataset['X_train'].shape[3]),
            'train_class_distribution': {},
            'val_class_distribution': {}
        }
        
        # Add class distributions
        for class_idx in range(len(self.classifier.CLASS_NAMES)):
            class_name = self.classifier.CLASS_NAMES[class_idx]
            train_count = int(np.sum(dataset['y_train'] == class_idx))
            val_count = int(np.sum(dataset['y_val'] == class_idx))
            
            metadata['train_class_distribution'][class_name] = train_count
            metadata['val_class_distribution'][class_name] = val_count
        
        # Save metadata
        metadata_file = self.output_dir / f'{dataset_type}_metadata_real.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved {metadata_file.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare training data from real satellite imagery'
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=2000,
        help='Number of samples per class (default: 2000)'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=64,
        help='Size of patches (default: 64)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=32,
        help='Stride for patch extraction (default: 32)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Fraction for training set (default: 0.8)'
    )
    parser.add_argument(
        '--processed-dir',
        type=Path,
        default=Path('data/processed'),
        help='Directory containing processed imagery (default: data/processed)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/training'),
        help='Output directory for training data (default: data/training)'
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("="*70)
    logger.info("Real Satellite Data Training Preparation")
    logger.info("="*70)
    
    try:
        # Create preparator
        preparator = RealDatasetPreparator(
            processed_dir=args.processed_dir,
            output_dir=args.output_dir
        )
        
        # Prepare CNN dataset
        dataset = preparator.prepare_cnn_dataset(
            patch_size=args.patch_size,
            stride=args.stride,
            samples_per_class=args.samples_per_class,
            train_split=args.train_split
        )
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING DATA PREPARATION SUMMARY")
        print("="*70)
        print(f"Data Source: Real Sentinel-2 Imagery")
        print(f"\nTraining Set:")
        print(f"  Samples: {len(dataset['X_train']):,}")
        print(f"  Shape: {dataset['X_train'].shape}")
        print(f"\nValidation Set:")
        print(f"  Samples: {len(dataset['X_val']):,}")
        print(f"  Shape: {dataset['X_val'].shape}")
        print(f"\nClasses: {len(preparator.classifier.CLASS_NAMES)}")
        for i, name in enumerate(preparator.classifier.CLASS_NAMES):
            train_count = np.sum(dataset['y_train'] == i)
            val_count = np.sum(dataset['y_val'] == i)
            print(f"  {i}. {name}: {train_count:,} train, {val_count:,} val")
        print(f"\nOutput directory: {args.output_dir}")
        print("="*70)
        
        logger.info("\n✅ Training data preparation complete!")
        logger.info("Ready to train CNN model on real data.")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
