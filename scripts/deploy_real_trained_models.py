#!/usr/bin/env python3
"""
Deployment Script for Real-Trained Models

This script deploys real-trained models to production by:
1. Backing up existing synthetic-trained models
2. Copying real-trained models to production location
3. Updating model registry with new metadata
4. Verifying models load correctly
5. Updating .env to enable AI predictions

Requirements: 5.4, 5.5, 6.4, 6.5, 9.5
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelDeploymentManager:
    """Manages deployment of real-trained models to production."""
    
    def __init__(self, models_dir: Path = Path("models")):
        """
        Initialize deployment manager.
        
        Args:
            models_dir: Directory containing models
        """
        self.models_dir = models_dir
        self.backup_dir = models_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.registry_file = models_dir / "model_registry.json"
        
        # Model configurations
        self.models = {
            'cnn': {
                'synthetic_model': 'crop_health_cnn.pth',
                'real_model': 'crop_health_cnn_real.pth',
                'synthetic_metrics': 'cnn_model_metrics.json',
                'real_metrics': 'cnn_model_metrics_real.json',
                'production_model': 'crop_health_cnn.pth',
                'production_metrics': 'cnn_model_metrics.json'
            },
            'lstm': {
                'synthetic_model': None,  # No synthetic LSTM model exists
                'real_model': 'crop_health_lstm_real.pth',
                'synthetic_metrics': None,
                'real_metrics': 'lstm_model_metrics_real.json',
                'production_model': 'crop_health_lstm.pth',
                'production_metrics': 'lstm_model_metrics.json'
            }
        }
    
    def backup_existing_models(self) -> Dict[str, bool]:
        """
        Backup existing synthetic-trained models.
        
        Returns:
            Dictionary with backup status for each model
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Backing up existing models")
        logger.info("=" * 80)
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created backup directory: {self.backup_dir}")
        
        backup_status = {}
        
        for model_type, config in self.models.items():
            logger.info(f"\nBacking up {model_type.upper()} model...")
            
            # Backup model file
            synthetic_model = config['synthetic_model']
            if synthetic_model:
                model_path = self.models_dir / synthetic_model
                if model_path.exists():
                    backup_path = self.backup_dir / synthetic_model
                    shutil.copy2(model_path, backup_path)
                    logger.info(f"  ✓ Backed up model: {synthetic_model} -> {backup_path}")
                    backup_status[f'{model_type}_model'] = True
                else:
                    logger.warning(f"  ⚠ Model not found: {model_path}")
                    backup_status[f'{model_type}_model'] = False
            else:
                logger.info(f"  - No synthetic {model_type.upper()} model to backup")
                backup_status[f'{model_type}_model'] = False
            
            # Backup metrics file
            synthetic_metrics = config['synthetic_metrics']
            if synthetic_metrics:
                metrics_path = self.models_dir / synthetic_metrics
                if metrics_path.exists():
                    backup_path = self.backup_dir / synthetic_metrics
                    shutil.copy2(metrics_path, backup_path)
                    logger.info(f"  ✓ Backed up metrics: {synthetic_metrics} -> {backup_path}")
                    backup_status[f'{model_type}_metrics'] = True
                else:
                    logger.warning(f"  ⚠ Metrics not found: {metrics_path}")
                    backup_status[f'{model_type}_metrics'] = False
            else:
                logger.info(f"  - No synthetic {model_type.upper()} metrics to backup")
                backup_status[f'{model_type}_metrics'] = False
        
        # Backup existing registry if it exists
        if self.registry_file.exists():
            backup_registry = self.backup_dir / "model_registry.json"
            shutil.copy2(self.registry_file, backup_registry)
            logger.info(f"\n✓ Backed up model registry: {backup_registry}")
            backup_status['registry'] = True
        else:
            logger.info("\n- No existing model registry to backup")
            backup_status['registry'] = False
        
        logger.info(f"\n✓ Backup completed: {self.backup_dir}")
        return backup_status
    
    def deploy_real_models(self) -> Dict[str, bool]:
        """
        Copy real-trained models to production location.
        
        Returns:
            Dictionary with deployment status for each model
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Deploying real-trained models")
        logger.info("=" * 80)
        
        deployment_status = {}
        
        for model_type, config in self.models.items():
            logger.info(f"\nDeploying {model_type.upper()} model...")
            
            # Check if real model exists
            real_model = config['real_model']
            real_model_path = self.models_dir / real_model
            
            if not real_model_path.exists():
                logger.error(f"  ✗ Real model not found: {real_model_path}")
                deployment_status[f'{model_type}_model'] = False
                continue
            
            # Copy real model to production name
            production_model = config['production_model']
            production_path = self.models_dir / production_model
            
            shutil.copy2(real_model_path, production_path)
            logger.info(f"  ✓ Deployed model: {real_model} -> {production_model}")
            deployment_status[f'{model_type}_model'] = True
            
            # Copy real metrics to production name
            real_metrics = config['real_metrics']
            real_metrics_path = self.models_dir / real_metrics
            
            if real_metrics_path.exists():
                production_metrics = config['production_metrics']
                production_metrics_path = self.models_dir / production_metrics
                
                shutil.copy2(real_metrics_path, production_metrics_path)
                logger.info(f"  ✓ Deployed metrics: {real_metrics} -> {production_metrics}")
                deployment_status[f'{model_type}_metrics'] = True
            else:
                logger.warning(f"  ⚠ Metrics not found: {real_metrics_path}")
                deployment_status[f'{model_type}_metrics'] = False
        
        logger.info("\n✓ Deployment completed")
        return deployment_status
    
    def update_model_registry(self) -> Dict[str, any]:
        """
        Update model registry with new metadata.
        
        Returns:
            Updated registry dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Updating model registry")
        logger.info("=" * 80)
        
        registry = {
            'last_updated': datetime.now().isoformat(),
            'deployment_type': 'real_trained_models',
            'models': {}
        }
        
        for model_type, config in self.models.items():
            logger.info(f"\nProcessing {model_type.upper()} model metadata...")
            
            # Load real model metrics
            real_metrics_path = self.models_dir / config['real_metrics']
            
            if not real_metrics_path.exists():
                logger.warning(f"  ⚠ Metrics not found: {real_metrics_path}")
                continue
            
            with open(real_metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Create registry entry
            registry['models'][model_type] = {
                'model_file': config['production_model'],
                'metrics_file': config['production_metrics'],
                'model_type': metrics.get('model_type', model_type.upper()),
                'framework': metrics.get('framework', 'PyTorch'),
                'version': metrics.get('version', '2.0'),
                'trained_on': metrics.get('trained_on', 'real_satellite_data'),
                'data_source': metrics.get('data_source', 'Sentinel-2 via Sentinel Hub API'),
                'training_date': metrics.get('training_date'),
                'deployed_date': datetime.now().isoformat(),
                'accuracy': metrics.get('metrics', {}).get('accuracy'),
                'status': 'active',
                'backup_location': str(self.backup_dir)
            }
            
            logger.info(f"  ✓ Added {model_type.upper()} to registry")
            logger.info(f"    - Trained on: {registry['models'][model_type]['trained_on']}")
            logger.info(f"    - Data source: {registry['models'][model_type]['data_source']}")
            logger.info(f"    - Accuracy: {registry['models'][model_type]['accuracy']}")
        
        # Save registry
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"\n✓ Registry updated: {self.registry_file}")
        return registry
    
    def verify_models_load(self) -> Dict[str, bool]:
        """
        Verify that deployed models load correctly.
        
        Returns:
            Dictionary with verification status for each model
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Verifying models load correctly")
        logger.info("=" * 80)
        
        verification_status = {}
        
        for model_type, config in self.models.items():
            logger.info(f"\nVerifying {model_type.upper()} model...")
            
            production_model = config['production_model']
            model_path = self.models_dir / production_model
            
            if not model_path.exists():
                logger.error(f"  ✗ Model not found: {model_path}")
                verification_status[model_type] = False
                continue
            
            try:
                # Try to load with PyTorch
                import torch
                
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Check if it's a state dict or full checkpoint
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        logger.info(f"  ✓ Loaded checkpoint with state_dict")
                    else:
                        state_dict = checkpoint
                        logger.info(f"  ✓ Loaded state_dict directly")
                    
                    # Count parameters
                    num_params = sum(p.numel() for p in state_dict.values())
                    logger.info(f"  ✓ Model has {num_params:,} parameters")
                else:
                    logger.info(f"  ✓ Loaded model object")
                
                # Verify metrics file
                metrics_file = config['production_metrics']
                metrics_path = self.models_dir / metrics_file
                
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    logger.info(f"  ✓ Metrics file loaded successfully")
                    logger.info(f"    - Trained on: {metrics.get('trained_on', 'N/A')}")
                    logger.info(f"    - Data source: {metrics.get('data_source', 'N/A')}")
                else:
                    logger.warning(f"  ⚠ Metrics file not found: {metrics_path}")
                
                verification_status[model_type] = True
                logger.info(f"  ✓ {model_type.upper()} model verification passed")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to load {model_type.upper()} model: {e}")
                verification_status[model_type] = False
        
        logger.info("\n✓ Model verification completed")
        return verification_status
    
    def update_env_file(self, env_path: Path = Path(".env")) -> bool:
        """
        Update .env file to enable AI predictions.
        
        Args:
            env_path: Path to .env file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Updating .env configuration")
        logger.info("=" * 80)
        
        if not env_path.exists():
            logger.error(f"✗ .env file not found: {env_path}")
            return False
        
        try:
            # Read current .env
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            # Update USE_AI_MODELS setting
            updated = False
            for i, line in enumerate(lines):
                if line.startswith('USE_AI_MODELS='):
                    old_value = line.strip()
                    lines[i] = 'USE_AI_MODELS=true\n'
                    new_value = lines[i].strip()
                    
                    if old_value != new_value:
                        logger.info(f"  Updated: {old_value} -> {new_value}")
                        updated = True
                    else:
                        logger.info(f"  Already set: {new_value}")
                    break
            else:
                # USE_AI_MODELS not found, add it
                lines.append('\n# AI Models Configuration (Updated by deployment script)\n')
                lines.append('USE_AI_MODELS=true\n')
                logger.info("  Added: USE_AI_MODELS=true")
                updated = True
            
            # Write updated .env
            with open(env_path, 'w') as f:
                f.writelines(lines)
            
            logger.info(f"✓ .env file updated: {env_path}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to update .env: {e}")
            return False
    
    def generate_deployment_report(
        self,
        backup_status: Dict[str, bool],
        deployment_status: Dict[str, bool],
        verification_status: Dict[str, bool],
        registry: Dict[str, any],
        env_updated: bool
    ) -> Dict[str, any]:
        """
        Generate comprehensive deployment report.
        
        Args:
            backup_status: Backup operation results
            deployment_status: Deployment operation results
            verification_status: Verification operation results
            registry: Updated model registry
            env_updated: Whether .env was updated
            
        Returns:
            Deployment report dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("DEPLOYMENT SUMMARY")
        logger.info("=" * 80)
        
        report = {
            'deployment_date': datetime.now().isoformat(),
            'backup_location': str(self.backup_dir),
            'backup_status': backup_status,
            'deployment_status': deployment_status,
            'verification_status': verification_status,
            'env_updated': env_updated,
            'models_deployed': list(registry.get('models', {}).keys()),
            'success': all(verification_status.values()) and env_updated
        }
        
        # Print summary
        logger.info("\n1. BACKUP STATUS:")
        for key, status in backup_status.items():
            symbol = "✓" if status else "✗"
            logger.info(f"   {symbol} {key}: {'Success' if status else 'Failed/Skipped'}")
        
        logger.info("\n2. DEPLOYMENT STATUS:")
        for key, status in deployment_status.items():
            symbol = "✓" if status else "✗"
            logger.info(f"   {symbol} {key}: {'Success' if status else 'Failed'}")
        
        logger.info("\n3. VERIFICATION STATUS:")
        for model_type, status in verification_status.items():
            symbol = "✓" if status else "✗"
            logger.info(f"   {symbol} {model_type.upper()}: {'Verified' if status else 'Failed'}")
        
        logger.info(f"\n4. ENVIRONMENT CONFIGURATION:")
        symbol = "✓" if env_updated else "✗"
        logger.info(f"   {symbol} .env updated: {'Yes' if env_updated else 'No'}")
        
        logger.info(f"\n5. MODELS DEPLOYED:")
        for model_type in report['models_deployed']:
            logger.info(f"   ✓ {model_type.upper()}")
        
        # Overall status
        logger.info("\n" + "=" * 80)
        if report['success']:
            logger.info("✓ DEPLOYMENT SUCCESSFUL")
            logger.info("  All models deployed and verified successfully!")
            logger.info("  AI predictions are now enabled with real-trained models.")
        else:
            logger.warning("⚠ DEPLOYMENT COMPLETED WITH WARNINGS")
            logger.warning("  Some operations failed. Please review the logs above.")
        logger.info("=" * 80)
        
        # Save report
        report_path = self.models_dir / "deployment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Deployment report saved: {report_path}")
        
        return report
    
    def deploy(self) -> Dict[str, any]:
        """
        Execute complete deployment pipeline.
        
        Returns:
            Deployment report
        """
        logger.info("=" * 80)
        logger.info("REAL-TRAINED MODELS DEPLOYMENT")
        logger.info("=" * 80)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Models directory: {self.models_dir.absolute()}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Backup existing models
            backup_status = self.backup_existing_models()
            
            # Step 2: Deploy real models
            deployment_status = self.deploy_real_models()
            
            # Step 3: Update model registry
            registry = self.update_model_registry()
            
            # Step 4: Verify models load
            verification_status = self.verify_models_load()
            
            # Step 5: Update .env
            env_updated = self.update_env_file()
            
            # Generate report
            report = self.generate_deployment_report(
                backup_status,
                deployment_status,
                verification_status,
                registry,
                env_updated
            )
            
            return report
            
        except Exception as e:
            logger.error(f"✗ Deployment failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    """Main entry point for deployment script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deploy real-trained models to production',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy models with default settings
  python scripts/deploy_real_trained_models.py
  
  # Deploy models from custom directory
  python scripts/deploy_real_trained_models.py --models-dir /path/to/models
  
  # Dry run (backup and verify only, no deployment)
  python scripts/deploy_real_trained_models.py --dry-run
        """
    )
    
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path('models'),
        help='Directory containing models (default: models)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform backup and verification only, no deployment'
    )
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = ModelDeploymentManager(models_dir=args.models_dir)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No deployment will be performed")
        logger.info("=" * 80)
        
        # Only backup and verify
        backup_status = manager.backup_existing_models()
        verification_status = manager.verify_models_load()
        
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUN COMPLETED")
        logger.info("=" * 80)
        
        return 0
    
    # Execute full deployment
    try:
        report = manager.deploy()
        
        if report['success']:
            logger.info("\n✓ Deployment completed successfully!")
            return 0
        else:
            logger.warning("\n⚠ Deployment completed with warnings. Check logs for details.")
            return 1
            
    except Exception as e:
        logger.error(f"\n✗ Deployment failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
