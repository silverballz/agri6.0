# Comprehensive Logging System

This document describes the comprehensive logging system implemented for the AgriFlux platform.

## Overview

The logging system provides centralized, structured logging capabilities across all components of the AgriFlux pipeline. It addresses all logging requirements (7.1-7.5) with specialized loggers for different use cases.

## Features

### 1. API Request/Response Logging (Requirement 7.1)

Detailed logging of all API interactions with Sentinel Hub:

- Request URL, method, headers, and payload
- Response status code, headers, and body
- Request duration
- Automatic sanitization of sensitive headers (tokens, API keys)
- Error details for failed requests

**Example:**
```python
from src.utils.logging_config import setup_logging, create_api_logger

logger = setup_logging('my_module', log_file='logs/my_module.log')
api_logger = create_api_logger(logger)

# Log request
api_logger.log_request(
    method='POST',
    url='https://api.example.com/endpoint',
    headers={'Authorization': 'Bearer token'},
    payload={'key': 'value'}
)

# Log response
api_logger.log_response(
    status_code=200,
    body='{"result": "success"}',
    duration=1.5
)
```

### 2. Download Progress Logging (Requirement 7.2)

Track download progress with detailed statistics:

- Total items and current progress
- Success/failure counts
- Download size and duration per item
- Overall statistics and success rate
- Human-readable size formatting

**Example:**
```python
from src.utils.logging_config import create_download_logger

download_logger = create_download_logger(logger)

# Start tracking
download_logger.start_download(total_items=20, description="imagery dates")

# Log each item
for item in items:
    download_logger.log_item_progress(
        item_name=item.name,
        success=True,
        size_bytes=15_000_000,
        duration=2.5
    )

# Get summary
summary = download_logger.finish_download()
```

### 3. Training Metrics Logging (Requirement 7.3)

Comprehensive logging of model training:

- Training configuration
- Epoch-by-epoch metrics (loss, accuracy, etc.)
- Best model checkpoints
- Training history
- Final evaluation metrics

**Example:**
```python
from src.utils.logging_config import create_training_logger

training_logger = create_training_logger(logger)

# Start training
training_logger.log_training_start(
    model_name='CropHealthCNN',
    total_epochs=50,
    config={'batch_size': 32, 'learning_rate': 0.001}
)

# Log each epoch
for epoch in range(epochs):
    training_logger.log_epoch_metrics(
        epoch=epoch,
        total_epochs=epochs,
        train_metrics={'loss': train_loss, 'accuracy': train_acc},
        val_metrics={'loss': val_loss, 'accuracy': val_acc},
        is_best=is_best
    )

# Complete training
summary = training_logger.log_training_complete(final_metrics)
```

### 4. Error Logging with Stack Traces (Requirement 7.4)

Detailed error logging with full context:

- Exception type and message
- Complete stack trace
- Context information
- Error categorization (fatal vs non-fatal)
- Error statistics and summary

**Example:**
```python
from src.utils.logging_config import create_error_logger

error_logger = create_error_logger(logger)

try:
    # Some operation
    risky_operation()
except Exception as e:
    error_logger.log_error(
        error=e,
        context={
            'operation': 'data_processing',
            'input_file': 'data.csv',
            'step': 'validation'
        },
        fatal=False
    )

# Get error summary
summary = error_logger.get_error_summary()
```

### 5. Pipeline Summary Reports (Requirement 7.5)

Generate comprehensive pipeline execution reports:

- Overall pipeline status
- Step-by-step results and timing
- Performance metrics
- Error summary
- JSON export for analysis

**Example:**
```python
from src.utils.logging_config import create_pipeline_reporter

reporter = create_pipeline_reporter(logger, 'Data Download Pipeline')

# Log each step
reporter.log_step_start('authentication', 'Authenticate with API')
# ... do work ...
reporter.log_step_complete('authentication', status='success', metrics={...})

reporter.log_step_start('download', 'Download imagery')
# ... do work ...
reporter.log_step_complete('download', status='success', metrics={...})

# Add pipeline metrics
reporter.add_metric('total_downloads', 20)
reporter.add_metric('success_rate', 0.95)

# Generate summary
summary = reporter.generate_summary(output_file='logs/pipeline_summary.json')
```

## Setup and Configuration

### Basic Setup

```python
from src.utils.logging_config import setup_logging

# Simple setup with defaults
logger = setup_logging('my_module')

# With log file
logger = setup_logging('my_module', log_file='logs/my_module.log')

# With custom configuration
logger = setup_logging(
    module_name='my_module',
    log_file='logs/my_module.log',
    level=logging.DEBUG,
    console=True,
    file_rotation=True,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5
)
```

### Log File Rotation

The logging system supports automatic log file rotation:

- Maximum file size: 10MB (configurable)
- Backup count: 5 files (configurable)
- Automatic compression of old logs
- Prevents disk space issues

### Thread Safety

All logging operations are thread-safe, making the system suitable for:

- Multi-threaded downloads
- Parallel data processing
- Concurrent API requests

## Integration with Existing Code

### Sentinel Hub Client

The API client already includes basic logging. To enhance it with the comprehensive system:

```python
from src.utils.logging_config import setup_logging, create_api_logger

logger = setup_logging('sentinel_hub_client', log_file='logs/api_requests.log')
api_logger = create_api_logger(logger)

# Use in request_with_retry method
api_logger.log_request(method, url, headers=headers, payload=payload)
response = requests.request(method, url, **kwargs)
api_logger.log_response(response.status_code, response.headers, response.text, duration)
```

### Download Scripts

Enhance download scripts with progress tracking:

```python
from src.utils.logging_config import setup_logging, create_download_logger

logger = setup_logging('download_script', log_file='logs/downloads.log')
download_logger = create_download_logger(logger)

download_logger.start_download(total_items=len(imagery_list), description="imagery dates")

for imagery in imagery_list:
    try:
        result = download_imagery(imagery)
        download_logger.log_item_progress(
            item_name=imagery.date,
            success=True,
            size_bytes=result.size,
            duration=result.duration
        )
    except Exception as e:
        download_logger.log_item_progress(
            item_name=imagery.date,
            success=False,
            error=str(e)
        )

summary = download_logger.finish_download()
```

### Training Scripts

Training scripts already have good logging. Enhance with structured metrics:

```python
from src.utils.logging_config import setup_logging, create_training_logger

logger = setup_logging('cnn_training', log_file='logs/cnn_training.log')
training_logger = create_training_logger(logger)

training_logger.log_training_start(model_name, epochs, config)

for epoch in range(epochs):
    # ... training code ...
    training_logger.log_epoch_metrics(
        epoch, epochs,
        train_metrics={'loss': train_loss, 'accuracy': train_acc},
        val_metrics={'loss': val_loss, 'accuracy': val_acc},
        is_best=val_acc > best_val_acc
    )

summary = training_logger.log_training_complete(final_metrics)
```

## Log File Organization

Recommended log file structure:

```
logs/
├── api_requests.log          # All API interactions
├── downloads.log             # Download progress
├── cnn_training.log          # CNN training metrics
├── lstm_training.log         # LSTM training metrics
├── data_processing.log       # Data processing operations
├── pipeline_orchestration.log # Complete pipeline runs
├── errors.log                # All errors across system
└── summaries/
    ├── pipeline_YYYYMMDD_HHMMSS.json
    └── training_YYYYMMDD_HHMMSS.json
```

## Best Practices

### 1. Use Appropriate Log Levels

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages for potentially problematic situations
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical errors that may cause system failure

### 2. Include Context

Always include relevant context when logging:

```python
logger.info(f"Processing imagery for date {date}, tile {tile_id}")
```

### 3. Log at Key Points

Log at important points in execution:

- Start and end of major operations
- Before and after API calls
- When errors occur
- When important decisions are made

### 4. Use Structured Logging

Use the specialized loggers for structured data:

```python
# Good - structured
training_logger.log_epoch_metrics(epoch, epochs, train_metrics, val_metrics)

# Less good - unstructured
logger.info(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}")
```

### 5. Generate Summaries

Always generate summaries for long-running operations:

```python
summary = download_logger.finish_download()
summary = training_logger.log_training_complete(metrics)
summary = reporter.generate_summary(output_file='logs/summary.json')
```

## Examples

See `examples/logging_system_example.py` for complete working examples of all logging features.

Run the examples:

```bash
python examples/logging_system_example.py
```

This will create example log files in the `logs/` directory demonstrating all logging capabilities.

## Troubleshooting

### Log Files Not Created

Ensure the logs directory exists:

```python
from pathlib import Path
Path('logs').mkdir(exist_ok=True)
```

### Duplicate Log Messages

If you see duplicate messages, check that you're not calling `setup_logging` multiple times for the same module. The function includes protection against this, but it's best to call it once per module.

### Log Files Too Large

If log files grow too large:

1. Enable file rotation (enabled by default)
2. Reduce the `max_bytes` parameter
3. Increase `backup_count` to keep more history
4. Use log level filtering to reduce verbosity

### Performance Impact

The logging system is designed to have minimal performance impact:

- Asynchronous file writes
- Efficient string formatting
- Conditional debug logging
- Thread-safe operations

For performance-critical sections, use conditional logging:

```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Expensive operation result: {expensive_computation()}")
```

## Requirements Mapping

This logging system fulfills all requirements:

- **7.1**: API request/response logging via `APIRequestLogger`
- **7.2**: Download progress logging via `DownloadProgressLogger`
- **7.3**: Training metrics logging via `TrainingMetricsLogger`
- **7.4**: Error logging with stack traces via `ErrorLogger`
- **7.5**: Pipeline summary reports via `PipelineSummaryReporter`

## Future Enhancements

Potential future improvements:

1. **Remote Logging**: Send logs to centralized logging service
2. **Log Analysis**: Automated log analysis and alerting
3. **Metrics Dashboard**: Real-time visualization of metrics
4. **Log Aggregation**: Combine logs from multiple pipeline runs
5. **Performance Profiling**: Integrated performance metrics
