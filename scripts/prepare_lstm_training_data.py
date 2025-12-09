#!/usr/bin/env python3
"""
Prepare LSTM Training Data from Real Satellite Imagery

This script prepares temporal training datasets from real Sentinel-2 imagery for LSTM model training:
1. Finds only real (non-synthetic) imagery directories
2. Sorts imagery by acquisition date
3. Creates sliding window sequences over time
4. Generates input sequences and target values
5. Splits into train/validation (80/20)
6. Saves prepared temporal data with metadata

Usage:
    python scripts/prepare_lstm_training_data.py --sequence-length 10 --samples 1000
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lstm_training_data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LSTMDatasetPreparator:
    """
    Prepare LSTM training datasets from real temporal satellite imagery.
    
    This class ensures that only real (non-synthetic) imagery is used for training,
    and creates temporal sequences for time-series prediction.
    """
    
    def __init__(self, processed_dir: Path, output_dir: Path):
        """
        Initialize LSTM dataset preparator.
        
        Args:
            processed_dir: Directory containing processed imagery
            output_dir: Directory for output training data
        """
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LSTMDatasetPreparator initialized")
        logger.info(f"Processed imagery directory: {self.processed_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _find_real_imagery_dirs(self) -> List[Tuple[Path, datetime]]:
        """
        Find all directories containing real (non-synthetic) imagery.
        
        Only includes directories where metadata.json has synthetic=false.
        
        Returns:
            List of tuples (path, acquisition_date), sorted by date
        """
        logger.info("Searching for real imagery directories...")
        
        real_imagery = []
        
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
                # Parse acquisition date
                acq_date_str = metadata.get('acquisition_date', '')
                try:
                    # Handle both date and datetime formats
                    if 'T' in acq_date_str:
                        acq_date = datetime.fromisoformat(acq_date_str.replace('Z', '+00:00'))
                    else:
                        acq_date = datetime.strptime(acq_date_str, '%Y-%m-%d')
                    
                    real_imagery.append((img_dir, acq_date))
                    logger.info(f"  ✓ Found real imagery: {img_dir.name} ({acq_date_str})")
                    
                except Exception as e:
                    logger.warning(f"  Skipping {img_dir.name} - invalid date format: {e}")
                    continue
            else:
                logger.debug(f"  Skipping {img_dir.name} - synthetic data")
        
        # Sort by acquisition date
        real_imagery = sorted(real_imagery, key=lambda x: x[1])
        
        logger.info(f"Found {len(real_imagery)} real imagery directories")
        
        if len(real_imagery) == 0:
            logger.warning("No real imagery found! Make sure to run download_real_satellite_data.py first")
        
        return real_imagery
    
    def _load_ndvi(self, img_dir: Path) -> np.ndarray:
        """
        Load NDVI data from imagery directory.
        
        Args:
            img_dir: Path to imagery directory
            
        Returns:
            NDVI array
        """
        ndvi_file = img_dir / 'NDVI.npy'
        if not ndvi_file.exists():
            raise FileNotFoundError(f"NDVI file not found: {ndvi_file}")
        
        ndvi = np.load(ndvi_file)
        return ndvi
    
    def _extract_spatial_samples(
        self,
        ndvi: np.ndarray,
        num_samples: int,
        sample_size: int = 32
    ) -> np.ndarray:
        """
        Extract random spatial samples from NDVI image.
        
        Args:
            ndvi: NDVI array [H, W]
            num_samples: Number of samples to extract
            sample_size: Size of each sample patch
            
        Returns:
            Array of samples [num_samples, sample_size, sample_size]
        """
        h, w = ndvi.shape
        
        if h < sample_size or w < sample_size:
            # If image is too small, just use mean value
            return np.full((num_samples, sample_size, sample_size), np.nanmean(ndvi))
        
        samples = []
        
        for _ in range(num_samples):
            # Random position
            i = np.random.randint(0, h - sample_size + 1)
            j = np.random.randint(0, w - sample_size + 1)
            
            # Extract patch
            patch = ndvi[i:i+sample_size, j:j+sample_size]
            
            # Skip patches with too many NaN values
            if np.sum(np.isnan(patch)) > 0.5 * sample_size * sample_size:
                # Try again with mean value
                patch = np.full((sample_size, sample_size), np.nanmean(ndvi))
            
            samples.append(patch)
        
        return np.array(samples)
    
    def prepare_lstm_dataset(
        self,
        sequence_length: int = 10,
        samples_per_sequence: int = 100,
        sample_size: int = 32,
        train_split: float = 0.8
    ) -> Dict[str, np.ndarray]:
        """
        Prepare LSTM training dataset from real temporal imagery.
        
        Creates sliding window sequences over time, where each sequence contains
        NDVI values from consecutive dates, and the target is the next time step.
        
        Args:
            sequence_length: Number of time steps in each sequence (default: 10)
            samples_per_sequence: Number of spatial samples per sequence (default: 100)
            sample_size: Size of spatial samples (default: 32x32)
            train_split: Fraction for training set (default: 0.8)
            
        Returns:
            Dictionary with X_train, y_train, X_val, y_val arrays
        """
        logger.info("="*70)
        logger.info("Preparing LSTM Training Dataset from Real Temporal Imagery")
        logger.info("="*70)
        logger.info(f"Parameters:")
        logger.info(f"  Sequence length: {sequence_length} time steps")
        logger.info(f"  Samples per sequence: {samples_per_sequence}")
        logger.info(f"  Sample size: {sample_size}x{sample_size}")
        logger.info(f"  Train/Val split: {train_split:.0%}/{1-train_split:.0%}")
        
        # Find all real imagery directories sorted by date
        imagery_list = self._find_real_imagery_dirs()
        
        if not imagery_list:
            raise ValueError(
                "No real imagery found! Please run download_real_satellite_data.py first."
            )
        
        if len(imagery_list) < sequence_length + 1:
            raise ValueError(
                f"Insufficient temporal data: need at least {sequence_length + 1} dates, "
                f"but only found {len(imagery_list)}. Please download more imagery."
            )
        
        logger.info(f"\nFound {len(imagery_list)} temporal imagery dates")
        logger.info(f"Date range: {imagery_list[0][1].strftime('%Y-%m-%d')} to "
                   f"{imagery_list[-1][1].strftime('%Y-%m-%d')}")
        
        # Extract temporal sequences
        logger.info(f"\nExtracting temporal sequences...")
        logger.info(f"  Possible sequences: {len(imagery_list) - sequence_length}")
        
        sequences = []
        targets = []
        sequence_dates = []
        
        for i in range(len(imagery_list) - sequence_length):
            logger.info(f"  Processing sequence {i+1}/{len(imagery_list) - sequence_length}")
            
            try:
                # Load NDVI for sequence
                seq_ndvi = []
                seq_date_strs = []
                
                for j in range(sequence_length):
                    img_dir, acq_date = imagery_list[i + j]
                    ndvi = self._load_ndvi(img_dir)
                    seq_ndvi.append(ndvi)
                    seq_date_strs.append(acq_date.strftime('%Y-%m-%d'))
                
                # Load target (next time step)
                target_dir, target_date = imagery_list[i + sequence_length]
                target_ndvi = self._load_ndvi(target_dir)
                
                # Extract spatial samples from each time step
                # This creates multiple training examples from each temporal sequence
                for sample_idx in range(samples_per_sequence):
                    # Get consistent spatial locations across time
                    h, w = seq_ndvi[0].shape
                    
                    if h < sample_size or w < sample_size:
                        # Use mean values if image too small
                        seq_samples = [np.nanmean(ndvi) for ndvi in seq_ndvi]
                        target_sample = np.nanmean(target_ndvi)
                    else:
                        # Random spatial location
                        i_pos = np.random.randint(0, h - sample_size + 1)
                        j_pos = np.random.randint(0, w - sample_size + 1)
                        
                        # Extract patches from same location across time
                        seq_samples = []
                        for ndvi in seq_ndvi:
                            patch = ndvi[i_pos:i_pos+sample_size, j_pos:j_pos+sample_size]
                            # Use mean of patch
                            seq_samples.append(np.nanmean(patch))
                        
                        # Extract target patch
                        target_patch = target_ndvi[i_pos:i_pos+sample_size, j_pos:j_pos+sample_size]
                        target_sample = np.nanmean(target_patch)
                    
                    # Skip if too many NaN values
                    if np.any(np.isnan(seq_samples)) or np.isnan(target_sample):
                        continue
                    
                    sequences.append(seq_samples)
                    targets.append(target_sample)
                    sequence_dates.append(seq_date_strs)
                
                logger.info(f"    Extracted {samples_per_sequence} samples from sequence")
                
            except Exception as e:
                logger.error(f"    Failed to process sequence: {e}")
                continue
        
        if not sequences:
            raise ValueError("Failed to extract any temporal sequences from real imagery")
        
        # Convert to numpy arrays
        logger.info("\nConverting to numpy arrays...")
        X_seq = np.array(sequences, dtype=np.float32)  # [num_sequences, sequence_length]
        y_target = np.array(targets, dtype=np.float32)  # [num_sequences]
        
        # Reshape for LSTM: [num_sequences, sequence_length, 1]
        X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)
        
        logger.info(f"  Total sequences: {len(X_seq):,}")
        logger.info(f"  Sequence shape: {X_seq.shape}")
        logger.info(f"  Target shape: {y_target.shape}")
        
        # Split into train/validation
        logger.info("\nSplitting into train/validation sets...")
        dataset = self._train_val_split(X_seq, y_target, train_split)
        
        # Save prepared data
        logger.info("\nSaving prepared training data...")
        self._save_training_data(dataset, 'lstm', sequence_dates[:10])
        
        logger.info("\n" + "="*70)
        logger.info("LSTM Dataset Preparation Complete!")
        logger.info("="*70)
        logger.info(f"Training sequences: {len(dataset['X_train']):,}")
        logger.info(f"Validation sequences: {len(dataset['X_val']):,}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*70)
        
        return dataset
    
    def _train_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train and validation sets.
        
        Args:
            X: Input sequences
            y: Target values
            train_split: Fraction for training set
            
        Returns:
            Dictionary with X_train, y_train, X_val, y_val
        """
        # Calculate split index
        n_train = int(len(X) * train_split)
        
        # Shuffle data before splitting
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        # Split data
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:]
        y_val = y[n_train:]
        
        logger.info(f"  Training set: {len(X_train):,} sequences ({train_split:.0%})")
        logger.info(f"  Validation set: {len(X_val):,} sequences ({1-train_split:.0%})")
        
        # Statistics
        logger.info(f"  Training target range: [{np.min(y_train):.4f}, {np.max(y_train):.4f}]")
        logger.info(f"  Training target mean: {np.mean(y_train):.4f} ± {np.std(y_train):.4f}")
        logger.info(f"  Validation target range: [{np.min(y_val):.4f}, {np.max(y_val):.4f}]")
        logger.info(f"  Validation target mean: {np.mean(y_val):.4f} ± {np.std(y_val):.4f}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }
    
    def _save_training_data(
        self,
        dataset: Dict[str, np.ndarray],
        dataset_type: str,
        sample_dates: List[List[str]]
    ):
        """
        Save prepared training data to disk with metadata.
        
        Args:
            dataset: Dictionary with train/val data
            dataset_type: Type of dataset ('lstm')
            sample_dates: Sample of sequence dates for metadata
        """
        # Save arrays
        for key, array in dataset.items():
            filename = f"{dataset_type}_{key}_real.npy"
            filepath = self.output_dir / filename
            np.save(filepath, array)
            logger.info(f"  Saved {filename}: shape={array.shape}, dtype={array.dtype}")
        
        # Create metadata
        metadata = {
            'dataset_type': dataset_type,
            'data_source': 'real',
            'created_at': datetime.now().isoformat(),
            'num_train_sequences': int(len(dataset['X_train'])),
            'num_val_sequences': int(len(dataset['X_val'])),
            'sequence_length': int(dataset['X_train'].shape[1]),
            'input_features': int(dataset['X_train'].shape[2]),
            'train_target_stats': {
                'min': float(np.min(dataset['y_train'])),
                'max': float(np.max(dataset['y_train'])),
                'mean': float(np.mean(dataset['y_train'])),
                'std': float(np.std(dataset['y_train']))
            },
            'val_target_stats': {
                'min': float(np.min(dataset['y_val'])),
                'max': float(np.max(dataset['y_val'])),
                'mean': float(np.mean(dataset['y_val'])),
                'std': float(np.std(dataset['y_val']))
            },
            'sample_sequence_dates': sample_dates
        }
        
        # Save metadata
        metadata_file = self.output_dir / f'{dataset_type}_metadata_real.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved {metadata_file.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare LSTM training data from real temporal satellite imagery'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=10,
        help='Number of time steps in each sequence (default: 10)'
    )
    parser.add_argument(
        '--samples-per-sequence',
        type=int,
        default=100,
        help='Number of spatial samples per temporal sequence (default: 100)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=32,
        help='Size of spatial samples (default: 32)'
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
    logger.info("Real Satellite Data LSTM Training Preparation")
    logger.info("="*70)
    
    try:
        # Create preparator
        preparator = LSTMDatasetPreparator(
            processed_dir=args.processed_dir,
            output_dir=args.output_dir
        )
        
        # Prepare LSTM dataset
        dataset = preparator.prepare_lstm_dataset(
            sequence_length=args.sequence_length,
            samples_per_sequence=args.samples_per_sequence,
            sample_size=args.sample_size,
            train_split=args.train_split
        )
        
        # Print summary
        print("\n" + "="*70)
        print("LSTM TRAINING DATA PREPARATION SUMMARY")
        print("="*70)
        print(f"Data Source: Real Sentinel-2 Temporal Imagery")
        print(f"\nTraining Set:")
        print(f"  Sequences: {len(dataset['X_train']):,}")
        print(f"  Shape: {dataset['X_train'].shape}")
        print(f"  Target range: [{np.min(dataset['y_train']):.4f}, {np.max(dataset['y_train']):.4f}]")
        print(f"\nValidation Set:")
        print(f"  Sequences: {len(dataset['X_val']):,}")
        print(f"  Shape: {dataset['X_val'].shape}")
        print(f"  Target range: [{np.min(dataset['y_val']):.4f}, {np.max(dataset['y_val']):.4f}]")
        print(f"\nSequence Length: {dataset['X_train'].shape[1]} time steps")
        print(f"Output directory: {args.output_dir}")
        print("="*70)
        
        logger.info("\n✅ LSTM training data preparation complete!")
        logger.info("Ready to train LSTM model on real temporal data.")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
