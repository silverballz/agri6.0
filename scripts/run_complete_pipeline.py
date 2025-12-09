#!/usr/bin/env python3
"""
Complete Real Satellite Data Pipeline Orchestration Script

This script orchestrates the complete end-to-end pipeline for downloading real
Sentinel-2 satellite imagery, preparing training datasets, and training AI models.

Pipeline Steps:
1. Download real satellite data from Sentinel Hub API
2. Validate data quality
3. Prepare CNN training dataset
4. Prepare LSTM training dataset
5. Train CNN model on real data
6. Train LSTM model on real data
7. Compare model performance (synthetic vs real)
8. Update .env to enable AI models
9. Generate summary report

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5

Usage:
    python scripts/run_complete_pipeline.py [--skip-download] [--skip-validation]
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_orchestration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Represents a single step in the pipeline."""
    step_number: int
    name: str
    description: str
    script: Optional[str]
    args: List[str]
    status: StepStatus
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    output_summary: Optional[Dict[str, Any]] = None


@dataclass
class PipelineReport:
    """Complete pipeline execution report."""
    pipeline_name: str
    start_time: str
    end_time: Optional[str]
    total_duration_seconds: Optional[float]
    steps: List[PipelineStep]
    overall_status: StepStatus
    summary_statistics: Dict[str, Any]


class PipelineOrchestrator:
    """
    Orchestrates the complete real satellite data pipeline.
    
    Implements requirements:
    - 9.1: Execute complete pipeline (download, prepare, train)
    - 9.2: Provide progress updates for each major step
    - 9.3: Halt execution and report failure point on errors
    - 9.4: Display summary statistics on completion
    - 9.5: Update .env to enable AI models
    """
    
    def __init__(self, skip_download: bool = False, skip_validation: bool = False):
        """
        Initialize pipeline orchestrator.
        
        Args:
            skip_download: Skip data download step (use existing data)
            skip_validation: Skip data validation step
        """
        self.skip_download = skip_download
        self.skip_validation = skip_validation
        self.start_time = datetime.now()
        
        # Define pipeline steps
        self.steps = self._define_pipeline_steps()
        
        logger.info("="*80)
        logger.info("Real Satellite Data Pipeline Orchestrator")
        logger.info("="*80)
        logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Skip download: {skip_download}")
        logger.info(f"Skip validation: {skip_validation}")
        logger.info(f"Total steps: {len(self.steps)}")
        logger.info("="*80)
    
    def _define_pipeline_steps(self) -> List[PipelineStep]:
        """
        Define all pipeline steps.
        
        Returns:
            List of PipelineStep objects
        """
        steps = [
            PipelineStep(
                step_number=1,
                name="Download Real Satellite Data",
                description="Download 15-20 real Sentinel-2 imagery dates from Sentinel Hub API",
                script="scripts/download_real_satellite_data.py",
                args=["--days-back", "365", "--target-count", "20", "--cloud-threshold", "20.0"],
                status=StepStatus.SKIPPED if self.skip_download else StepStatus.PENDING
            ),
            PipelineStep(
                step_number=2,
                name="Validate Data Quality",
                description="Validate downloaded imagery meets quality requirements",
                script="scripts/validate_data_quality.py",
                args=[],
                status=StepStatus.SKIPPED if self.skip_validation else StepStatus.PENDING
            ),
            PipelineStep(
                step_number=3,
                name="Prepare CNN Training Data",
                description="Extract patches and create balanced dataset for CNN training",
                script="scripts/prepare_real_training_data.py",
                args=["--samples-per-class", "2000"],
                status=StepStatus.PENDING
            ),
            PipelineStep(
                step_number=4,
                name="Prepare LSTM Training Data",
                description="Create temporal sequences for LSTM training",
                script="scripts/prepare_lstm_training_data.py",
                args=["--sequence-length", "10"],
                status=StepStatus.PENDING
            ),
            PipelineStep(
                step_number=5,
                name="Train CNN Model",
                description="Train CNN model on real satellite imagery",
                script="scripts/train_cnn_on_real_data.py",
                args=["--epochs", "50", "--patience", "10", "--min-accuracy", "0.85"],
                status=StepStatus.PENDING
            ),
            PipelineStep(
                step_number=6,
                name="Train LSTM Model",
                description="Train LSTM model on real temporal sequences",
                script="scripts/train_lstm_on_real_data.py",
                args=["--epochs", "100", "--patience", "15", "--min-accuracy", "0.80"],
                status=StepStatus.PENDING
            ),
            PipelineStep(
                step_number=7,
                name="Compare Model Performance",
                description="Compare synthetic-trained vs real-trained models",
                script="scripts/compare_model_performance.py",
                args=[],
                status=StepStatus.PENDING
            ),
            PipelineStep(
                step_number=8,
                name="Update Configuration",
                description="Update .env file to enable AI models",
                script=None,  # Built-in function
                args=[],
                status=StepStatus.PENDING
            )
        ]
        
        return steps
    
    def run(self) -> PipelineReport:
        """
        Execute the complete pipeline (Requirement 9.1).
        
        Returns:
            PipelineReport with execution results
        """
        logger.info("\n" + "="*80)
        logger.info("Starting Pipeline Execution")
        logger.info("="*80 + "\n")
        
        # Execute each step
        for step in self.steps:
            if step.status == StepStatus.SKIPPED:
                logger.info(f"Step {step.step_number}: {step.name} - SKIPPED")
                continue
            
            # Progress update (Requirement 9.2)
            self._report_step_progress(step)
            
            # Execute step
            success = self._execute_step(step)
            
            # Check for failure (Requirement 9.3)
            if not success:
                logger.error(f"\n{'='*80}")
                logger.error(f"Pipeline FAILED at Step {step.step_number}: {step.name}")
                logger.error(f"Error: {step.error_message}")
                logger.error(f"{'='*80}\n")
                
                # Generate failure report
                return self._generate_report(overall_status=StepStatus.FAILED)
        
        # All steps completed successfully
        logger.info("\n" + "="*80)
        logger.info("Pipeline Execution Complete!")
        logger.info("="*80 + "\n")
        
        # Generate success report (Requirement 9.4)
        return self._generate_report(overall_status=StepStatus.SUCCESS)
    
    def _report_step_progress(self, step: PipelineStep):
        """
        Report progress for a pipeline step (Requirement 9.2).
        
        Args:
            step: PipelineStep to report on
        """
        logger.info("="*80)
        logger.info(f"Step {step.step_number}/{len(self.steps)}: {step.name}")
        logger.info("="*80)
        logger.info(f"Description: {step.description}")
        if step.script:
            logger.info(f"Script: {step.script}")
            if step.args:
                logger.info(f"Arguments: {' '.join(step.args)}")
        logger.info(f"Status: {step.status.value}")
        logger.info("="*80 + "\n")

    def _execute_step(self, step: PipelineStep) -> bool:
        """
        Execute a single pipeline step.
        
        Args:
            step: PipelineStep to execute
            
        Returns:
            True if step succeeded, False otherwise
        """
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now().isoformat()
        start_timestamp = datetime.now()
        
        try:
            if step.script:
                # Execute Python script
                success = self._run_script(step)
            else:
                # Built-in function (e.g., update .env)
                if step.step_number == 8:
                    success = self._update_env_file(step)
                else:
                    raise ValueError(f"Unknown built-in step: {step.name}")
            
            # Update step status
            end_timestamp = datetime.now()
            step.end_time = end_timestamp.isoformat()
            step.duration_seconds = (end_timestamp - start_timestamp).total_seconds()
            
            if success:
                step.status = StepStatus.SUCCESS
                logger.info(f"\n✓ Step {step.step_number} completed successfully "
                          f"({step.duration_seconds:.1f}s)\n")
            else:
                step.status = StepStatus.FAILED
                logger.error(f"\n✗ Step {step.step_number} failed "
                           f"({step.duration_seconds:.1f}s)\n")
            
            return success
            
        except Exception as e:
            # Handle unexpected errors
            end_timestamp = datetime.now()
            step.end_time = end_timestamp.isoformat()
            step.duration_seconds = (end_timestamp - start_timestamp).total_seconds()
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            
            logger.error(f"\n✗ Step {step.step_number} failed with exception: {e}\n",
                        exc_info=True)
            
            return False
    
    def _run_script(self, step: PipelineStep) -> bool:
        """
        Run a Python script as a subprocess.
        
        Args:
            step: PipelineStep with script to run
            
        Returns:
            True if script succeeded (exit code 0), False otherwise
        """
        # Build command
        cmd = [sys.executable, step.script] + step.args
        
        logger.info(f"Executing: {' '.join(cmd)}\n")
        
        try:
            # Run script and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Log output
            if result.stdout:
                logger.info("Script output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            
            if result.stderr:
                logger.warning("Script errors:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.warning(f"  {line}")
            
            # Check exit code
            if result.returncode == 0:
                return True
            else:
                step.error_message = f"Script exited with code {result.returncode}"
                if result.stderr:
                    step.error_message += f": {result.stderr[:500]}"
                return False
                
        except Exception as e:
            step.error_message = f"Failed to execute script: {str(e)}"
            logger.error(step.error_message, exc_info=True)
            return False
    
    def _update_env_file(self, step: PipelineStep) -> bool:
        """
        Update .env file to enable AI models (Requirement 9.5).
        
        Args:
            step: PipelineStep for this operation
            
        Returns:
            True if update succeeded, False otherwise
        """
        logger.info("Updating .env configuration to enable AI models...")
        
        env_path = Path('.env')
        
        if not env_path.exists():
            step.error_message = ".env file not found"
            logger.error(f"  {step.error_message}")
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
                    updated = True
                    logger.info(f"  Updated: {old_value} -> USE_AI_MODELS=true")
                    break
            
            if not updated:
                # Add setting if not found
                lines.append('\n# AI Models\nUSE_AI_MODELS=true\n')
                logger.info("  Added: USE_AI_MODELS=true")
            
            # Write updated .env
            with open(env_path, 'w') as f:
                f.writelines(lines)
            
            logger.info("  ✓ .env file updated successfully")
            
            step.output_summary = {
                'file': str(env_path),
                'setting': 'USE_AI_MODELS=true',
                'updated': updated
            }
            
            return True
            
        except Exception as e:
            step.error_message = f"Failed to update .env: {str(e)}"
            logger.error(f"  {step.error_message}", exc_info=True)
            return False

    def _generate_report(self, overall_status: StepStatus) -> PipelineReport:
        """
        Generate pipeline execution report (Requirement 9.4).
        
        Args:
            overall_status: Overall pipeline status
            
        Returns:
            PipelineReport with execution summary
        """
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics()
        
        # Create report
        report = PipelineReport(
            pipeline_name="Real Satellite Data Integration Pipeline",
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=total_duration,
            steps=self.steps,
            overall_status=overall_status,
            summary_statistics=summary_stats
        )
        
        # Log summary
        self._log_summary(report)
        
        # Save report to file
        self._save_report(report)
        
        return report
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for the pipeline execution.
        
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'total_steps': len(self.steps),
            'completed_steps': sum(1 for s in self.steps if s.status == StepStatus.SUCCESS),
            'failed_steps': sum(1 for s in self.steps if s.status == StepStatus.FAILED),
            'skipped_steps': sum(1 for s in self.steps if s.status == StepStatus.SKIPPED),
            'total_duration_seconds': sum(
                s.duration_seconds for s in self.steps 
                if s.duration_seconds is not None
            ),
            'step_durations': {
                s.name: s.duration_seconds 
                for s in self.steps 
                if s.duration_seconds is not None
            }
        }
        
        # Try to extract model performance metrics
        try:
            # Check for CNN model metrics
            cnn_metrics_path = Path('models/cnn_model_metrics_real.json')
            if cnn_metrics_path.exists():
                with open(cnn_metrics_path, 'r') as f:
                    cnn_metrics = json.load(f)
                    stats['cnn_accuracy'] = cnn_metrics['metrics']['accuracy']
            
            # Check for LSTM model metrics
            lstm_metrics_path = Path('models/lstm_model_metrics_real.json')
            if lstm_metrics_path.exists():
                with open(lstm_metrics_path, 'r') as f:
                    lstm_metrics = json.load(f)
                    stats['lstm_accuracy'] = lstm_metrics['metrics']['accuracy']
            
            # Check for comparison report
            comparison_path = Path('reports/model_comparison_report.json')
            if comparison_path.exists():
                with open(comparison_path, 'r') as f:
                    comparison = json.load(f)
                    stats['model_comparison'] = comparison
        except Exception as e:
            logger.warning(f"Could not extract model metrics: {e}")
        
        return stats

    def _log_summary(self, report: PipelineReport):
        """
        Log pipeline execution summary (Requirement 9.4).
        
        Args:
            report: PipelineReport to summarize
        """
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Pipeline: {report.pipeline_name}")
        logger.info(f"Start time: {report.start_time}")
        logger.info(f"End time: {report.end_time}")
        logger.info(f"Total duration: {report.total_duration_seconds:.1f} seconds "
                   f"({report.total_duration_seconds/60:.1f} minutes)")
        logger.info(f"Overall status: {report.overall_status.value.upper()}")
        logger.info("")
        logger.info("Step Summary:")
        logger.info(f"  Total steps: {report.summary_statistics['total_steps']}")
        logger.info(f"  Completed: {report.summary_statistics['completed_steps']}")
        logger.info(f"  Failed: {report.summary_statistics['failed_steps']}")
        logger.info(f"  Skipped: {report.summary_statistics['skipped_steps']}")
        logger.info("")
        
        # Log individual step results
        logger.info("Step Details:")
        for step in report.steps:
            status_symbol = {
                StepStatus.SUCCESS: "✓",
                StepStatus.FAILED: "✗",
                StepStatus.SKIPPED: "⊘",
                StepStatus.PENDING: "○",
                StepStatus.RUNNING: "⟳"
            }.get(step.status, "?")
            
            duration_str = f"({step.duration_seconds:.1f}s)" if step.duration_seconds else ""
            logger.info(f"  {status_symbol} Step {step.step_number}: {step.name} "
                       f"{step.status.value.upper()} {duration_str}")
            
            if step.error_message:
                logger.info(f"      Error: {step.error_message}")
        
        logger.info("")
        
        # Log model performance if available
        if 'cnn_accuracy' in report.summary_statistics:
            logger.info("Model Performance:")
            logger.info(f"  CNN Accuracy: {report.summary_statistics['cnn_accuracy']:.4f}")
        
        if 'lstm_accuracy' in report.summary_statistics:
            logger.info(f"  LSTM Accuracy: {report.summary_statistics['lstm_accuracy']:.4f}")
        
        if 'model_comparison' in report.summary_statistics:
            logger.info("")
            logger.info("Model Comparison (Real vs Synthetic):")
            comparison = report.summary_statistics['model_comparison']
            if 'cnn_comparison' in comparison:
                cnn_comp = comparison['cnn_comparison']
                logger.info(f"  CNN Improvement: "
                          f"{cnn_comp.get('accuracy_improvement', 0):.2%}")
            if 'lstm_comparison' in comparison:
                lstm_comp = comparison['lstm_comparison']
                logger.info(f"  LSTM Improvement: "
                          f"{lstm_comp.get('mae_improvement', 0):.2%}")
        
        logger.info("")
        logger.info("="*80)
        
        # Final status message
        if report.overall_status == StepStatus.SUCCESS:
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("All models trained on real satellite data and ready for deployment.")
        else:
            logger.error("❌ PIPELINE FAILED!")
            logger.error("Please review the error messages above and fix issues before retrying.")
        
        logger.info("="*80 + "\n")
    
    def _save_report(self, report: PipelineReport):
        """
        Save pipeline report to JSON file.
        
        Args:
            report: PipelineReport to save
        """
        # Create reports directory
        reports_dir = Path('logs')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f'pipeline_report_{timestamp}.json'
        
        # Convert report to dictionary
        report_dict = {
            'pipeline_name': report.pipeline_name,
            'start_time': report.start_time,
            'end_time': report.end_time,
            'total_duration_seconds': report.total_duration_seconds,
            'overall_status': report.overall_status.value,
            'summary_statistics': report.summary_statistics,
            'steps': [
                {
                    'step_number': s.step_number,
                    'name': s.name,
                    'description': s.description,
                    'script': s.script,
                    'args': s.args,
                    'status': s.status.value,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'duration_seconds': s.duration_seconds,
                    'error_message': s.error_message,
                    'output_summary': s.output_summary
                }
                for s in report.steps
            ]
        }
        
        # Save to file
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Pipeline report saved to: {report_path}")


def main():
    """Main entry point for pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description='Run complete real satellite data pipeline'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download step (use existing data)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            skip_download=args.skip_download,
            skip_validation=args.skip_validation
        )
        
        # Run pipeline
        report = orchestrator.run()
        
        # Exit with appropriate code
        if report.overall_status == StepStatus.SUCCESS:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nFatal error in pipeline orchestration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
