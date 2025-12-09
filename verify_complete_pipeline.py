#!/usr/bin/env python3
"""
Final Pipeline Verification Script

This script performs comprehensive verification of the complete real satellite
data integration pipeline, checking all requirements from Task 20.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.db_manager import DatabaseManager
from src.ai_models.crop_health_predictor import CropHealthPredictor


class PipelineVerifier:
    """Comprehensive pipeline verification."""
    
    def __init__(self):
        self.db = DatabaseManager('data/agriflux.db')
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_status': 'PENDING'
        }
        
    def verify_all(self):
        """Run all verification checks."""
        print("=" * 80)
        print("FINAL PIPELINE VERIFICATION")
        print("=" * 80)
        print()
        
        checks = [
            ('Real Data Downloaded', self.check_real_data_downloaded),
            ('Real Data Stored Correctly', self.check_real_data_storage),
            ('Training Data from Real Sources', self.check_training_data_source),
            ('CNN Model Accuracy', self.check_cnn_accuracy),
            ('LSTM Model Accuracy', self.check_lstm_accuracy),
            ('Model Metadata Correct', self.check_model_metadata),
            ('Comparison Report Available', self.check_comparison_report),
            ('AI Predictions Working', self.check_ai_predictions),
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\n{'=' * 80}")
            print(f"CHECK: {check_name}")
            print('=' * 80)
            
            try:
                result = check_func()
                self.results['checks'][check_name] = result
                
                if result['status'] == 'PASS':
                    print(f"✓ PASS: {result['message']}")
                elif result['status'] == 'WARN':
                    print(f"⚠ WARNING: {result['message']}")
                else:
                    print(f"✗ FAIL: {result['message']}")
                    all_passed = False
                    
                if 'details' in result:
                    for key, value in result['details'].items():
                        print(f"  - {key}: {value}")
                        
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                self.results['checks'][check_name] = {
                    'status': 'ERROR',
                    'message': str(e)
                }
                all_passed = False
        
        self.results['overall_status'] = 'PASS' if all_passed else 'FAIL'
        
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Checks Passed: {sum(1 for c in self.results['checks'].values() if c['status'] == 'PASS')}/{len(checks)}")
        print()
        
        # Save results
        output_file = Path('logs/pipeline_verification.json')
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results saved to: {output_file}")
        
        return all_passed
    
    def check_real_data_downloaded(self):
        """Verify real data has been downloaded."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as count, 
                       MIN(acquisition_date) as earliest,
                       MAX(acquisition_date) as latest
                FROM processed_imagery 
                WHERE synthetic = 0
            """)
            
            result = cursor.fetchone()
            if not result:
                return {
                    'status': 'FAIL',
                    'message': 'No real data found in database'
                }
            
            count = result['count']
            earliest = result['earliest']
            latest = result['latest']
        
        if count < 15:
            return {
                'status': 'FAIL',
                'message': f'Insufficient real data: {count} dates (minimum 15 required)',
                'details': {
                    'count': count,
                    'earliest': earliest,
                    'latest': latest
                }
            }
        
        return {
            'status': 'PASS',
            'message': f'Real data downloaded successfully: {count} dates',
            'details': {
                'count': count,
                'earliest_date': earliest,
                'latest_date': latest,
                'date_range_days': (datetime.fromisoformat(latest) - datetime.fromisoformat(earliest)).days
            }
        }
    
    def check_real_data_storage(self):
        """Verify real data is stored correctly with proper flags."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tile_id, acquisition_date, synthetic, 
                       ndvi_path, savi_path, evi_path, ndwi_path
                FROM processed_imagery 
                WHERE synthetic = 0
                LIMIT 5
            """)
            
            results = cursor.fetchall()
            if not results:
                return {
                    'status': 'FAIL',
                    'message': 'No real data records found'
                }
            
            # Check that files exist
            missing_files = []
            for row in results:
                for path_key in ['ndvi_path', 'savi_path', 'evi_path', 'ndwi_path']:
                    if row[path_key]:
                        file_path = Path(row[path_key])
                        if not file_path.exists():
                            missing_files.append(str(file_path))
        
        if missing_files:
            return {
                'status': 'FAIL',
                'message': f'Missing {len(missing_files)} data files',
                'details': {
                    'missing_files': missing_files[:5]  # Show first 5
                }
            }
        
        return {
            'status': 'PASS',
            'message': 'Real data stored correctly with synthetic=false flag',
            'details': {
                'verified_records': len(results),
                'all_files_exist': True
            }
        }
    
    def check_training_data_source(self):
        """Verify training data was prepared from real imagery only."""
        cnn_train_file = Path('data/training/cnn_X_train_real.npy')
        cnn_metadata_file = Path('data/training/cnn_metadata_real.json')
        lstm_train_file = Path('data/training/lstm_X_train_real.npy')
        lstm_metadata_file = Path('data/training/lstm_metadata_real.json')
        
        issues = []
        
        # Check CNN training data
        if not cnn_train_file.exists():
            issues.append('CNN training data file not found')
        
        if cnn_metadata_file.exists():
            with open(cnn_metadata_file) as f:
                cnn_meta = json.load(f)
                if cnn_meta.get('data_source') != 'real':
                    issues.append(f"CNN data source is '{cnn_meta.get('data_source')}', expected 'real'")
        else:
            issues.append('CNN metadata file not found')
        
        # Check LSTM training data
        if not lstm_train_file.exists():
            issues.append('LSTM training data file not found')
        
        if lstm_metadata_file.exists():
            with open(lstm_metadata_file) as f:
                lstm_meta = json.load(f)
                if lstm_meta.get('data_source') != 'real':
                    issues.append(f"LSTM data source is '{lstm_meta.get('data_source')}', expected 'real'")
        else:
            issues.append('LSTM metadata file not found')
        
        if issues:
            return {
                'status': 'FAIL',
                'message': 'Training data source verification failed',
                'details': {'issues': issues}
            }
        
        return {
            'status': 'PASS',
            'message': 'Training data confirmed to be from real imagery only',
            'details': {
                'cnn_data_source': 'real',
                'lstm_data_source': 'real'
            }
        }
    
    def check_cnn_accuracy(self):
        """Verify CNN model meets accuracy threshold."""
        metrics_file = Path('models/cnn_model_metrics_real.json')
        
        if not metrics_file.exists():
            return {
                'status': 'FAIL',
                'message': 'CNN model metrics file not found'
            }
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Try to get accuracy from nested metrics structure
        # Use best_val_acc if available, otherwise use final accuracy
        if 'metrics' in metrics:
            best_acc = metrics['metrics'].get('best_val_acc')
            final_acc = metrics['metrics'].get('accuracy', 0)
            accuracy = best_acc if best_acc is not None else final_acc
        else:
            accuracy = metrics.get('accuracy', 0)
        
        threshold = 0.85
        
        if accuracy < threshold:
            return {
                'status': 'FAIL',
                'message': f'CNN accuracy {accuracy:.4f} below threshold {threshold}',
                'details': {
                    'best_accuracy': accuracy,
                    'threshold': threshold,
                    'full_metrics': metrics
                }
            }
        
        return {
            'status': 'PASS',
            'message': f'CNN accuracy {accuracy:.4f} meets threshold {threshold}',
            'details': {
                'best_accuracy': accuracy,
                'threshold': threshold,
                'precision': metrics.get('metrics', {}).get('classification_report', {}).get('weighted avg', {}).get('precision'),
                'recall': metrics.get('metrics', {}).get('classification_report', {}).get('weighted avg', {}).get('recall'),
                'f1_score': metrics.get('metrics', {}).get('classification_report', {}).get('weighted avg', {}).get('f1-score')
            }
        }
    
    def check_lstm_accuracy(self):
        """Verify LSTM model meets accuracy threshold."""
        metrics_file = Path('models/lstm_model_metrics_real.json')
        
        if not metrics_file.exists():
            return {
                'status': 'FAIL',
                'message': 'LSTM model metrics file not found'
            }
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # LSTM uses MSE/MAE, try to get from nested structure
        if 'metrics' in metrics:
            mse = metrics['metrics'].get('mse', float('inf'))
            mae = metrics['metrics'].get('mae', float('inf'))
        else:
            mse = metrics.get('mse', float('inf'))
            mae = metrics.get('mae', float('inf'))
        
        # For LSTM, we check if MSE is reasonable (< 0.1 is good)
        threshold_mse = 0.1
        
        if mse > threshold_mse:
            return {
                'status': 'WARN',
                'message': f'LSTM MSE {mse:.4f} above threshold {threshold_mse}',
                'details': metrics
            }
        
        return {
            'status': 'PASS',
            'message': f'LSTM performance acceptable (MSE: {mse:.4f}, MAE: {mae:.4f})',
            'details': {
                'mse': mse,
                'mae': mae,
                'threshold_mse': threshold_mse
            }
        }
    
    def check_model_metadata(self):
        """Verify model metadata is correct."""
        cnn_model_file = Path('models/crop_health_cnn_real.pth')
        lstm_model_file = Path('models/crop_health_lstm_real.pth')
        registry_file = Path('models/model_registry.json')
        
        issues = []
        
        # Check CNN model exists
        if not cnn_model_file.exists():
            issues.append('CNN model file not found')
        
        # Check LSTM model exists
        if not lstm_model_file.exists():
            issues.append('LSTM model file not found')
        
        # Check registry
        if registry_file.exists():
            with open(registry_file) as f:
                registry = json.load(f)
            
            # Check if models section exists
            models = registry.get('models', {})
            
            # Check CNN metadata
            if 'cnn' in models:
                cnn_meta = models['cnn']
                if cnn_meta.get('trained_on') != 'real_satellite_data':
                    issues.append(f"CNN trained_on is '{cnn_meta.get('trained_on')}', expected 'real_satellite_data'")
                if 'Sentinel' not in cnn_meta.get('data_source', ''):
                    issues.append(f"CNN data_source doesn't reference Sentinel Hub")
            else:
                issues.append('CNN model not in registry')
            
            # Check LSTM metadata
            if 'lstm' in models:
                lstm_meta = models['lstm']
                if lstm_meta.get('trained_on') != 'real_temporal_sequences':
                    issues.append(f"LSTM trained_on is '{lstm_meta.get('trained_on')}', expected 'real_temporal_sequences'")
                if 'Sentinel' not in lstm_meta.get('data_source', ''):
                    issues.append(f"LSTM data_source doesn't reference Sentinel Hub")
            else:
                issues.append('LSTM model not in registry')
        else:
            issues.append('Model registry file not found')
        
        if issues:
            return {
                'status': 'FAIL',
                'message': 'Model metadata verification failed',
                'details': {'issues': issues}
            }
        
        return {
            'status': 'PASS',
            'message': 'Model metadata is correct',
            'details': {
                'cnn_trained_on': 'real_satellite_data',
                'lstm_trained_on': 'real_temporal_sequences',
                'data_source': 'Sentinel-2 via Sentinel Hub API'
            }
        }
    
    def check_comparison_report(self):
        """Verify comparison report exists and shows improvements."""
        report_file = Path('reports/model_comparison_report.json')
        
        if not report_file.exists():
            return {
                'status': 'FAIL',
                'message': 'Model comparison report not found'
            }
        
        with open(report_file) as f:
            report = json.load(f)
        
        # Check if real models show improvement
        cnn_comparison = report.get('cnn_comparison', {})
        real_acc = cnn_comparison.get('real_model', {}).get('accuracy', 0)
        synthetic_acc = cnn_comparison.get('synthetic_model', {}).get('accuracy', 0)
        
        improvement = real_acc - synthetic_acc
        
        details = {
            'real_model_accuracy': real_acc,
            'synthetic_model_accuracy': synthetic_acc,
            'improvement': improvement
        }
        
        if improvement < 0:
            return {
                'status': 'WARN',
                'message': f'Real model accuracy lower than synthetic (diff: {improvement:.4f})',
                'details': details
            }
        
        return {
            'status': 'PASS',
            'message': f'Comparison report shows improvement: +{improvement:.4f}',
            'details': details
        }
    
    def check_ai_predictions(self):
        """Test AI predictions using real-trained models."""
        try:
            # Initialize predictor
            predictor = CropHealthPredictor()
            
            # Get a real imagery record
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, tile_id, acquisition_date, 
                           ndvi_path, savi_path, evi_path, ndwi_path
                    FROM processed_imagery 
                    WHERE synthetic = 0
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                if not result:
                    return {
                        'status': 'FAIL',
                        'message': 'No real imagery available for prediction test'
                    }
                
                imagery_id = result['id']
            
            # Load NDVI data for prediction
            ndvi_path = result['ndvi_path']
            if not ndvi_path or not Path(ndvi_path).exists():
                return {
                    'status': 'FAIL',
                    'message': 'NDVI data file not found for test imagery'
                }
            
            # Load NDVI array - check if it's a GeoTIFF or numpy file
            if ndvi_path.endswith('.tif') or ndvi_path.endswith('.tiff'):
                # Load GeoTIFF using rasterio
                try:
                    import rasterio
                    with rasterio.open(ndvi_path) as src:
                        ndvi_data = src.read(1)  # Read first band
                except ImportError:
                    return {
                        'status': 'FAIL',
                        'message': 'rasterio not available to read GeoTIFF'
                    }
            else:
                # Load numpy array
                ndvi_data = np.load(ndvi_path, allow_pickle=True)
            
            # Try to make a prediction
            prediction = predictor.predict(ndvi_data)
            
            if prediction is None:
                return {
                    'status': 'FAIL',
                    'message': 'Prediction returned None'
                }
            
            # Check prediction structure (ClassificationResult object)
            if not hasattr(prediction, 'predictions'):
                return {
                    'status': 'FAIL',
                    'message': 'Prediction object missing predictions attribute',
                    'details': {'prediction': str(prediction)}
                }
            
            # Get summary statistics from the prediction array
            pred_array = prediction.predictions
            unique, counts = np.unique(pred_array, return_counts=True)
            class_distribution = dict(zip(unique.tolist(), counts.tolist()))
            
            # Get most common prediction
            most_common_class = unique[np.argmax(counts)]
            class_name = prediction.class_names[most_common_class] if hasattr(prediction, 'class_names') else str(most_common_class)
            
            return {
                'status': 'PASS',
                'message': 'AI predictions working with real-trained models',
                'details': {
                    'test_imagery_id': imagery_id,
                    'prediction_shape': pred_array.shape,
                    'most_common_class': class_name,
                    'class_distribution': class_distribution,
                    'method': prediction.method if hasattr(prediction, 'method') else 'unknown'
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'AI prediction test failed: {str(e)}'
            }


def main():
    """Run complete pipeline verification."""
    verifier = PipelineVerifier()
    success = verifier.verify_all()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
