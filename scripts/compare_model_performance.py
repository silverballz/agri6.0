#!/usr/bin/env python3
"""
Model Performance Comparison Script

This script compares the performance of synthetic-trained and real-trained models
by evaluating both on the same test set. It generates comprehensive comparison
metrics including accuracy, precision, recall, F1 scores, and confusion matrices.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5

Usage:
    python scripts/compare_model_performance.py [--output-dir reports]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CropHealthCNN(nn.Module):
    """CNN model architecture (must match training script)."""
    
    def __init__(self, num_classes: int = 4):
        super(CropHealthCNN, self).__init__()
        
        # Convolutional block 1
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Convolutional block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Convolutional block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(model_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_data)
    """
    logger.info(f"Loading model from {model_path}...")
    
    # Create model
    model = CropHealthCNN(num_classes=4).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"  ✓ Model loaded successfully")
    
    return model, checkpoint


def load_test_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test dataset (using validation set as test set).
    
    Args:
        data_dir: Directory containing training data
        
    Returns:
        Tuple of (X_test, y_test)
    """
    logger.info(f"Loading test data from {data_dir}...")
    
    # Use validation set as test set for comparison
    X_test = np.load(data_dir / 'cnn_X_val_real.npy')
    y_test = np.load(data_dir / 'cnn_y_val_real.npy')
    
    logger.info(f"  X_test shape: {X_test.shape}")
    logger.info(f"  y_test shape: {y_test.shape}")
    
    return X_test, y_test


def evaluate_model_on_test_set(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """
    Evaluate model on test set and calculate comprehensive metrics (Requirement 10.2).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    model.eval()
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2).to(device)
    
    # Get predictions
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics (Requirement 10.2)
    accuracy = accuracy_score(y_test, all_preds)
    precision = precision_score(y_test, all_preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, all_preds)
    
    # Classification report
    class_report = classification_report(
        y_test, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Mean confidence
    mean_confidence = np.max(all_probs, axis=1).mean()
    
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Mean Confidence: {mean_confidence:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'mean_confidence': float(mean_confidence),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist()
    }


def compare_models(
    synthetic_metrics: Dict,
    real_metrics: Dict,
    class_names: List[str]
) -> Dict:
    """
    Compare performance between synthetic and real trained models (Requirement 10.3).
    
    Args:
        synthetic_metrics: Metrics from synthetic-trained model
        real_metrics: Metrics from real-trained model
        class_names: List of class names
        
    Returns:
        Dictionary of comparison results
    """
    logger.info("="*70)
    logger.info("Comparing Model Performance")
    logger.info("="*70)
    
    # Calculate improvements (Requirement 10.3)
    accuracy_improvement = real_metrics['accuracy'] - synthetic_metrics['accuracy']
    precision_improvement = real_metrics['precision'] - synthetic_metrics['precision']
    recall_improvement = real_metrics['recall'] - synthetic_metrics['recall']
    f1_improvement = real_metrics['f1_score'] - synthetic_metrics['f1_score']
    
    # Per-class improvements
    per_class_improvements = {}
    for i, class_name in enumerate(class_names):
        per_class_improvements[class_name] = {
            'precision_improvement': real_metrics['precision_per_class'][i] - synthetic_metrics['precision_per_class'][i],
            'recall_improvement': real_metrics['recall_per_class'][i] - synthetic_metrics['recall_per_class'][i],
            'f1_improvement': real_metrics['f1_per_class'][i] - synthetic_metrics['f1_per_class'][i]
        }
    
    # Identify which classes improved most (Requirement 10.3)
    best_improved_class = max(
        per_class_improvements.items(),
        key=lambda x: x[1]['f1_improvement']
    )
    
    logger.info(f"\nOverall Improvements:")
    logger.info(f"  Accuracy: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%)")
    logger.info(f"  Precision: {precision_improvement:+.4f} ({precision_improvement*100:+.2f}%)")
    logger.info(f"  Recall: {recall_improvement:+.4f} ({recall_improvement*100:+.2f}%)")
    logger.info(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
    
    logger.info(f"\nMost Improved Class: {best_improved_class[0]}")
    logger.info(f"  F1 Improvement: {best_improved_class[1]['f1_improvement']:+.4f}")
    
    logger.info(f"\nPer-Class Improvements:")
    for class_name, improvements in per_class_improvements.items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {improvements['precision_improvement']:+.4f}")
        logger.info(f"    Recall: {improvements['recall_improvement']:+.4f}")
        logger.info(f"    F1: {improvements['f1_improvement']:+.4f}")
    
    comparison = {
        'overall_improvements': {
            'accuracy': float(accuracy_improvement),
            'precision': float(precision_improvement),
            'recall': float(recall_improvement),
            'f1_score': float(f1_improvement)
        },
        'per_class_improvements': per_class_improvements,
        'best_improved_class': {
            'name': best_improved_class[0],
            'improvements': best_improved_class[1]
        },
        'synthetic_metrics': {
            'accuracy': synthetic_metrics['accuracy'],
            'precision': synthetic_metrics['precision'],
            'recall': synthetic_metrics['recall'],
            'f1_score': synthetic_metrics['f1_score']
        },
        'real_metrics': {
            'accuracy': real_metrics['accuracy'],
            'precision': real_metrics['precision'],
            'recall': real_metrics['recall'],
            'f1_score': real_metrics['f1_score']
        }
    }
    
    return comparison



def plot_confusion_matrix_comparison(
    synthetic_cm: np.ndarray,
    real_cm: np.ndarray,
    class_names: List[str],
    output_dir: Path
) -> None:
    """
    Create side-by-side confusion matrix visualizations (Requirement 10.4).
    
    Args:
        synthetic_cm: Confusion matrix from synthetic model
        real_cm: Confusion matrix from real model
        class_names: List of class names
        output_dir: Directory to save plots
    """
    logger.info("Creating confusion matrix comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Synthetic model confusion matrix
    sns.heatmap(
        synthetic_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('Synthetic-Trained Model\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Real model confusion matrix
    sns.heatmap(
        real_cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={'label': 'Count'}
    )
    axes[1].set_title('Real-Trained Model\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'confusion_matrix_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved confusion matrix comparison to: {output_path}")
    plt.close()


def plot_metrics_comparison(
    comparison: Dict,
    class_names: List[str],
    output_dir: Path
) -> None:
    """
    Create bar chart comparing metrics (Requirement 10.4).
    
    Args:
        comparison: Comparison results dictionary
        class_names: List of class names
        output_dir: Directory to save plots
    """
    logger.info("Creating metrics comparison plots...")
    
    # Overall metrics comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    synthetic_values = [comparison['synthetic_metrics'][m] for m in metrics]
    real_values = [comparison['real_metrics'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, synthetic_values, width, label='Synthetic-Trained', color='skyblue')
    bars2 = ax.bar(x + width/2, real_values, width, label='Real-Trained', color='lightgreen')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"  ✓ Saved metrics comparison to: {output_path}")
    plt.close()
    
    # Per-class F1 score comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    synthetic_f1 = [comparison['synthetic_metrics']['precision'], 
                    comparison['synthetic_metrics']['recall']]
    real_f1 = [comparison['real_metrics']['precision'],
               comparison['real_metrics']['recall']]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    # Get per-class F1 scores from classification reports
    synthetic_per_class = []
    real_per_class = []
    
    for class_name in class_names:
        # These are already in the metrics from evaluate_model_on_test_set
        pass
    
    plt.tight_layout()


def save_comparison_report(
    comparison: Dict,
    synthetic_metrics: Dict,
    real_metrics: Dict,
    output_dir: Path
) -> None:
    """
    Save comprehensive comparison report to JSON (Requirement 10.5).
    
    Args:
        comparison: Comparison results
        synthetic_metrics: Full metrics from synthetic model
        real_metrics: Full metrics from real model
        output_dir: Directory to save report
    """
    logger.info("="*70)
    logger.info("Saving Comparison Report")
    logger.info("="*70)
    
    report = {
        'comparison_date': datetime.now().isoformat(),
        'summary': {
            'overall_improvements': comparison['overall_improvements'],
            'best_improved_class': comparison['best_improved_class'],
            'per_class_improvements': comparison['per_class_improvements']
        },
        'synthetic_model': {
            'model_type': 'CNN',
            'training_data': 'synthetic',
            'metrics': {
                'accuracy': synthetic_metrics['accuracy'],
                'precision': synthetic_metrics['precision'],
                'recall': synthetic_metrics['recall'],
                'f1_score': synthetic_metrics['f1_score'],
                'mean_confidence': synthetic_metrics['mean_confidence'],
                'confusion_matrix': synthetic_metrics['confusion_matrix'],
                'classification_report': synthetic_metrics['classification_report']
            }
        },
        'real_model': {
            'model_type': 'CNN',
            'training_data': 'real_satellite_data',
            'data_source': 'Sentinel-2 via Sentinel Hub API',
            'metrics': {
                'accuracy': real_metrics['accuracy'],
                'precision': real_metrics['precision'],
                'recall': real_metrics['recall'],
                'f1_score': real_metrics['f1_score'],
                'mean_confidence': real_metrics['mean_confidence'],
                'confusion_matrix': real_metrics['confusion_matrix'],
                'classification_report': real_metrics['classification_report']
            }
        }
    }
    
    # Save report
    report_path = output_dir / 'model_comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"  ✓ Saved comparison report to: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\nSynthetic-Trained Model:")
    print(f"  Accuracy:  {synthetic_metrics['accuracy']:.4f}")
    print(f"  Precision: {synthetic_metrics['precision']:.4f}")
    print(f"  Recall:    {synthetic_metrics['recall']:.4f}")
    print(f"  F1 Score:  {synthetic_metrics['f1_score']:.4f}")
    
    print(f"\nReal-Trained Model:")
    print(f"  Accuracy:  {real_metrics['accuracy']:.4f}")
    print(f"  Precision: {real_metrics['precision']:.4f}")
    print(f"  Recall:    {real_metrics['recall']:.4f}")
    print(f"  F1 Score:  {real_metrics['f1_score']:.4f}")
    
    print(f"\nImprovements:")
    print(f"  Accuracy:  {comparison['overall_improvements']['accuracy']:+.4f} ({comparison['overall_improvements']['accuracy']*100:+.2f}%)")
    print(f"  Precision: {comparison['overall_improvements']['precision']:+.4f} ({comparison['overall_improvements']['precision']*100:+.2f}%)")
    print(f"  Recall:    {comparison['overall_improvements']['recall']:+.4f} ({comparison['overall_improvements']['recall']*100:+.2f}%)")
    print(f"  F1 Score:  {comparison['overall_improvements']['f1_score']:+.4f} ({comparison['overall_improvements']['f1_score']*100:+.2f}%)")
    
    print(f"\nMost Improved Class: {comparison['best_improved_class']['name']}")
    print(f"  F1 Improvement: {comparison['best_improved_class']['improvements']['f1_improvement']:+.4f}")
    
    print(f"\nReport saved to: {report_path}")
    print("="*70)



def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare synthetic-trained and real-trained model performance'
    )
    parser.add_argument(
        '--synthetic-model',
        type=Path,
        default=Path('models/crop_health_cnn.pth'),
        help='Path to synthetic-trained model (default: models/crop_health_cnn.pth)'
    )
    parser.add_argument(
        '--real-model',
        type=Path,
        default=Path('models/crop_health_cnn_real.pth'),
        help='Path to real-trained model (default: models/crop_health_cnn_real.pth)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/training'),
        help='Directory containing test data (default: data/training)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports'),
        help='Directory to save comparison reports (default: reports)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("Model Performance Comparison")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Synthetic model: {args.synthetic_model}")
    logger.info(f"Real model: {args.real_model}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        device = torch.device('cpu')
        class_names = ['Healthy', 'Moderate', 'Stressed', 'Critical']
        
        # Load test data (Requirement 10.1)
        X_test, y_test = load_test_data(args.data_dir)
        
        # Load synthetic-trained model (Requirement 10.1)
        logger.info("\n" + "="*70)
        logger.info("Loading Synthetic-Trained Model")
        logger.info("="*70)
        synthetic_model, synthetic_checkpoint = load_model(args.synthetic_model, device)
        
        # Load real-trained model (Requirement 10.1)
        logger.info("\n" + "="*70)
        logger.info("Loading Real-Trained Model")
        logger.info("="*70)
        real_model, real_checkpoint = load_model(args.real_model, device)
        
        # Evaluate synthetic model (Requirement 10.1, 10.2)
        logger.info("\n" + "="*70)
        logger.info("Evaluating Synthetic-Trained Model")
        logger.info("="*70)
        synthetic_metrics = evaluate_model_on_test_set(
            synthetic_model, X_test, y_test, device, class_names
        )
        
        # Evaluate real model (Requirement 10.1, 10.2)
        logger.info("\n" + "="*70)
        logger.info("Evaluating Real-Trained Model")
        logger.info("="*70)
        real_metrics = evaluate_model_on_test_set(
            real_model, X_test, y_test, device, class_names
        )
        
        # Compare models (Requirement 10.3)
        comparison = compare_models(synthetic_metrics, real_metrics, class_names)
        
        # Create visualizations (Requirement 10.4)
        logger.info("\n" + "="*70)
        logger.info("Creating Visualizations")
        logger.info("="*70)
        plot_confusion_matrix_comparison(
            np.array(synthetic_metrics['confusion_matrix']),
            np.array(real_metrics['confusion_matrix']),
            class_names,
            args.output_dir
        )
        plot_metrics_comparison(comparison, class_names, args.output_dir)
        
        # Save comparison report (Requirement 10.5)
        save_comparison_report(
            comparison,
            synthetic_metrics,
            real_metrics,
            args.output_dir
        )
        
        logger.info(f"\n✅ Comparison complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Fatal error during comparison: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
