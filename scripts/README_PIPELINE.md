# Complete Pipeline Orchestration Script

## Overview

The `run_complete_pipeline.py` script orchestrates the complete end-to-end pipeline for integrating real Sentinel-2 satellite data into the AgriFlux system. It automates all steps from data download to model training and deployment.

## Requirements

This script implements the following requirements from the design specification:

- **9.1**: Execute complete pipeline (download, prepare datasets, train models)
- **9.2**: Provide progress updates for each major step
- **9.3**: Halt execution and report failure point on errors
- **9.4**: Display summary statistics on completion
- **9.5**: Update .env to enable AI models

## Pipeline Steps

The pipeline executes the following steps in order:

1. **Download Real Satellite Data** - Downloads 15-20 Sentinel-2 imagery dates from Sentinel Hub API
2. **Validate Data Quality** - Validates downloaded imagery meets quality requirements
3. **Prepare CNN Training Data** - Extracts patches and creates balanced dataset for CNN
4. **Prepare LSTM Training Data** - Creates temporal sequences for LSTM training
5. **Train CNN Model** - Trains CNN model on real satellite imagery
6. **Train LSTM Model** - Trains LSTM model on real temporal sequences
7. **Compare Model Performance** - Compares synthetic-trained vs real-trained models
8. **Update Configuration** - Updates .env file to enable AI models

## Usage

### Basic Usage

Run the complete pipeline with default settings:

```bash
python scripts/run_complete_pipeline.py
```

### Skip Data Download

If you already have real satellite data downloaded, skip the download step:

```bash
python scripts/run_complete_pipeline.py --skip-download
```

### Skip Data Validation

Skip the data quality validation step:

```bash
python scripts/run_complete_pipeline.py --skip-validation
```

### Combined Options

Skip both download and validation:

```bash
python scripts/run_complete_pipeline.py --skip-download --skip-validation
```

## Output

### Progress Reporting

The script provides detailed progress updates for each step:

```
================================================================================
Step 1/8: Download Real Satellite Data
================================================================================
Description: Download 15-20 real Sentinel-2 imagery dates from Sentinel Hub API
Script: scripts/download_real_satellite_data.py
Arguments: --days-back 365 --target-count 20 --cloud-threshold 20.0
Status: pending
================================================================================
```

### Error Handling

If any step fails, the pipeline halts immediately and reports the failure:

```
================================================================================
Pipeline FAILED at Step 3: Prepare CNN Training Data
Error: No real imagery found! Please run download_real_satellite_data.py first.
================================================================================
```

### Summary Report

Upon completion, the script generates a comprehensive summary:

```
================================================================================
PIPELINE EXECUTION SUMMARY
================================================================================
Pipeline: Real Satellite Data Integration Pipeline
Start time: 2024-12-09T10:30:00
End time: 2024-12-09T12:45:00
Total duration: 8100.0 seconds (135.0 minutes)
Overall status: SUCCESS

Step Summary:
  Total steps: 8
  Completed: 8
  Failed: 0
  Skipped: 0

Step Details:
  ✓ Step 1: Download Real Satellite Data SUCCESS (1200.0s)
  ✓ Step 2: Validate Data Quality SUCCESS (60.0s)
  ✓ Step 3: Prepare CNN Training Data SUCCESS (300.0s)
  ✓ Step 4: Prepare LSTM Training Data SUCCESS (180.0s)
  ✓ Step 5: Train CNN Model SUCCESS (3600.0s)
  ✓ Step 6: Train LSTM Model SUCCESS (2400.0s)
  ✓ Step 7: Compare Model Performance SUCCESS (120.0s)
  ✓ Step 8: Update Configuration SUCCESS (1.0s)

Model Performance:
  CNN Accuracy: 0.8723
  LSTM Accuracy: 0.8156

Model Comparison (Real vs Synthetic):
  CNN Improvement: 12.45%
  LSTM Improvement: 8.32%

================================================================================
✅ PIPELINE COMPLETED SUCCESSFULLY!
All models trained on real satellite data and ready for deployment.
================================================================================
```

## Logs

The pipeline generates detailed logs in two locations:

1. **Console Output** - Real-time progress and status updates
2. **Log File** - `logs/pipeline_orchestration.log` - Complete execution log
3. **Pipeline Report** - `logs/pipeline_report_TIMESTAMP.json` - JSON report with all details

## Pipeline Report JSON

The pipeline saves a detailed JSON report with the following structure:

```json
{
  "pipeline_name": "Real Satellite Data Integration Pipeline",
  "start_time": "2024-12-09T10:30:00",
  "end_time": "2024-12-09T12:45:00",
  "total_duration_seconds": 8100.0,
  "overall_status": "success",
  "summary_statistics": {
    "total_steps": 8,
    "completed_steps": 8,
    "failed_steps": 0,
    "skipped_steps": 0,
    "cnn_accuracy": 0.8723,
    "lstm_accuracy": 0.8156,
    "model_comparison": { ... }
  },
  "steps": [
    {
      "step_number": 1,
      "name": "Download Real Satellite Data",
      "status": "success",
      "duration_seconds": 1200.0,
      ...
    },
    ...
  ]
}
```

## Error Recovery

If the pipeline fails:

1. **Review the error message** - The pipeline will report which step failed and why
2. **Check the logs** - Review `logs/pipeline_orchestration.log` for detailed error information
3. **Fix the issue** - Address the root cause (e.g., API credentials, missing data)
4. **Resume execution** - Use `--skip-download` or `--skip-validation` to skip completed steps

## Exit Codes

- **0** - Pipeline completed successfully
- **1** - Pipeline failed at some step
- **130** - Pipeline interrupted by user (Ctrl+C)

## Prerequisites

Before running the pipeline, ensure:

1. **Sentinel Hub API credentials** are configured in `.env`:
   ```
   SENTINEL_HUB_CLIENT_ID=your_client_id
   SENTINEL_HUB_CLIENT_SECRET=your_client_secret
   SENTINEL_HUB_INSTANCE_ID=your_instance_id
   ```

2. **Required directories exist**:
   - `data/processed/` - For processed imagery
   - `data/training/` - For training datasets
   - `models/` - For trained models
   - `logs/` - For log files

3. **Python dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

## Estimated Execution Time

Typical execution times for each step:

- Download Real Satellite Data: 20-60 minutes (depends on network and API)
- Validate Data Quality: 1-2 minutes
- Prepare CNN Training Data: 5-10 minutes
- Prepare LSTM Training Data: 3-5 minutes
- Train CNN Model: 30-90 minutes (depends on hardware)
- Train LSTM Model: 40-120 minutes (depends on hardware)
- Compare Model Performance: 2-5 minutes
- Update Configuration: < 1 second

**Total estimated time: 2-5 hours**

## Troubleshooting

### Common Issues

1. **API Authentication Errors**
   - Verify Sentinel Hub credentials in `.env`
   - Check that credentials are valid and not expired

2. **No Imagery Found**
   - Adjust `--days-back` parameter to search longer time period
   - Increase `--cloud-threshold` to allow more cloudy imagery

3. **Insufficient Training Data**
   - Download more imagery dates (increase `--target-count`)
   - Adjust patch extraction parameters

4. **Model Training Fails**
   - Check available memory (models require ~4GB RAM)
   - Reduce batch size if out of memory
   - Increase training epochs if accuracy is low

5. **Permission Errors**
   - Ensure write permissions for `data/`, `models/`, and `logs/` directories
   - Run with appropriate user permissions

## Advanced Configuration

To customize pipeline behavior, edit the step definitions in `run_complete_pipeline.py`:

```python
PipelineStep(
    step_number=1,
    name="Download Real Satellite Data",
    script="scripts/download_real_satellite_data.py",
    args=["--days-back", "365", "--target-count", "20"],  # Modify these
    status=StepStatus.PENDING
)
```

## Integration with Existing Workflows

The pipeline can be integrated into CI/CD workflows:

```yaml
# Example GitHub Actions workflow
- name: Run Real Data Pipeline
  run: python scripts/run_complete_pipeline.py --skip-download
  timeout-minutes: 300
```

## Support

For issues or questions:

1. Check the logs in `logs/pipeline_orchestration.log`
2. Review the pipeline report JSON
3. Consult the individual script documentation
4. Check the main project README

## Related Scripts

- `download_real_satellite_data.py` - Download satellite imagery
- `validate_data_quality.py` - Validate data quality
- `prepare_real_training_data.py` - Prepare CNN training data
- `prepare_lstm_training_data.py` - Prepare LSTM training data
- `train_cnn_on_real_data.py` - Train CNN model
- `train_lstm_on_real_data.py` - Train LSTM model
- `compare_model_performance.py` - Compare model performance
