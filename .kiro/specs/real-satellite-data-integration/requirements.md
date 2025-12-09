# Requirements Document

## Introduction

The AgriFlux system currently uses synthetic satellite data for training AI models because the Sentinel Hub API integration has critical bugs preventing real data download. This feature will fix the API integration, download real multi-temporal Sentinel-2 imagery for the Ludhiana region, and retrain all AI models on actual satellite data to ensure production-ready accuracy.

## Glossary

- **Sentinel Hub API**: Commercial API service providing access to Sentinel-2 satellite imagery
- **Sentinel-2**: European Space Agency satellite constellation providing multispectral imagery
- **SAFE Format**: Standard Archive Format for Europe - native Sentinel-2 data format
- **Vegetation Indices**: Calculated metrics (NDVI, SAVI, EVI, NDWI) derived from satellite bands
- **Multi-temporal Dataset**: Collection of satellite images from different dates for time-series analysis
- **Ludhiana Region**: Agricultural area in Punjab, India (30.9-31.0°N, 75.8-75.9°E)
- **CNN Model**: Convolutional Neural Network for spatial crop health classification
- **LSTM Model**: Long Short-Term Memory network for temporal trend prediction
- **Synthetic Data**: Artificially generated data used as fallback when real data unavailable
- **GeoTIFF**: Georeferenced raster image format for satellite data storage

## Requirements

### Requirement 1

**User Story:** As a system administrator, I want the Sentinel Hub API integration to successfully download real satellite imagery, so that the system operates on actual data instead of synthetic fallbacks.

#### Acceptance Criteria

1. WHEN the system queries the Sentinel Hub API with valid credentials THEN the API SHALL return available imagery metadata without 406 errors
2. WHEN the system requests imagery for a valid date range THEN the API SHALL use current or past dates only
3. WHEN the API request format is constructed THEN the system SHALL comply with Sentinel Hub API v1 specifications
4. WHEN authentication occurs THEN the system SHALL obtain and refresh OAuth2 tokens correctly
5. WHEN API rate limits are encountered THEN the system SHALL implement exponential backoff retry logic

### Requirement 2

**User Story:** As a data scientist, I want to download 15-20 real Sentinel-2 imagery dates for the Ludhiana region, so that I have sufficient temporal data for training time-series models.

#### Acceptance Criteria

1. WHEN querying for imagery THEN the system SHALL search the last 365 days from the current date
2. WHEN filtering imagery THEN the system SHALL select only images with cloud coverage below 20%
3. WHEN downloading bands THEN the system SHALL retrieve B02, B03, B04, and B08 at 10m resolution
4. WHEN processing imagery THEN the system SHALL calculate NDVI, SAVI, EVI, and NDWI indices
5. WHEN saving processed data THEN the system SHALL store both GeoTIFF files and numpy arrays with metadata marking them as real data

### Requirement 3

**User Story:** As a database administrator, I want all downloaded real satellite imagery to be properly stored in the database, so that the system can query and retrieve actual imagery records.

#### Acceptance Criteria

1. WHEN imagery is successfully downloaded THEN the system SHALL insert a record into the processed_imagery table
2. WHEN storing imagery metadata THEN the system SHALL include acquisition date, tile ID, cloud coverage, and file paths
3. WHEN marking data provenance THEN the system SHALL set synthetic flag to false for real data
4. WHEN querying imagery THEN the system SHALL distinguish between real and synthetic data sources
5. WHEN retrieving latest imagery THEN the system SHALL prioritize real data over synthetic data

### Requirement 4

**User Story:** As a machine learning engineer, I want to prepare training datasets from real satellite imagery, so that models learn from actual agricultural patterns.

#### Acceptance Criteria

1. WHEN extracting training patches THEN the system SHALL create 64x64 pixel samples from real imagery
2. WHEN generating labels THEN the system SHALL use rule-based classification on real NDVI values
3. WHEN balancing datasets THEN the system SHALL ensure equal representation of all crop health classes
4. WHEN splitting data THEN the system SHALL allocate 80% for training and 20% for validation
5. WHEN saving training data THEN the system SHALL store numpy arrays with metadata indicating real data source

### Requirement 5

**User Story:** As a machine learning engineer, I want to retrain the CNN model on real satellite data, so that spatial crop health predictions are accurate for production use.

#### Acceptance Criteria

1. WHEN training the CNN THEN the system SHALL use real satellite imagery patches as input
2. WHEN training completes THEN the model SHALL achieve minimum 85% validation accuracy on real data
3. WHEN evaluating performance THEN the system SHALL generate confusion matrix and classification report
4. WHEN saving the model THEN the system SHALL store PyTorch weights with metadata indicating training on real data
5. WHEN model accuracy is below threshold THEN the system SHALL log warnings and request more training data

### Requirement 6

**User Story:** As a machine learning engineer, I want to retrain the LSTM model on real temporal data, so that trend predictions reflect actual agricultural cycles.

#### Acceptance Criteria

1. WHEN preparing time-series data THEN the system SHALL use real multi-temporal imagery sequences
2. WHEN training the LSTM THEN the system SHALL learn temporal patterns from actual vegetation index changes
3. WHEN training completes THEN the model SHALL achieve minimum 80% validation accuracy on real sequences
4. WHEN evaluating performance THEN the system SHALL measure prediction accuracy across different time horizons
5. WHEN saving the model THEN the system SHALL store PyTorch weights with metadata indicating training on real data

### Requirement 7

**User Story:** As a system administrator, I want comprehensive logging of the data download and training pipeline, so that I can debug issues and verify data provenance.

#### Acceptance Criteria

1. WHEN API requests occur THEN the system SHALL log request URLs, parameters, and response status codes
2. WHEN imagery is downloaded THEN the system SHALL log acquisition dates, cloud coverage, and file sizes
3. WHEN training progresses THEN the system SHALL log epoch metrics, loss values, and accuracy scores
4. WHEN errors occur THEN the system SHALL log detailed error messages with stack traces
5. WHEN pipeline completes THEN the system SHALL generate summary report with data statistics and model metrics

### Requirement 8

**User Story:** As a developer, I want validation scripts to verify real data quality, so that I can ensure downloaded imagery meets requirements before training.

#### Acceptance Criteria

1. WHEN validating imagery THEN the system SHALL check that all required bands are present
2. WHEN checking data quality THEN the system SHALL verify vegetation indices are within valid ranges
3. WHEN verifying temporal coverage THEN the system SHALL confirm minimum 15 dates are available
4. WHEN inspecting metadata THEN the system SHALL validate that synthetic flag is false for real data
5. WHEN quality checks fail THEN the system SHALL report specific issues and prevent training on invalid data

### Requirement 9

**User Story:** As a system operator, I want a single command to execute the complete pipeline, so that I can easily download data and retrain models.

#### Acceptance Criteria

1. WHEN executing the pipeline script THEN the system SHALL download real imagery, prepare datasets, and train both models
2. WHEN the pipeline runs THEN the system SHALL provide progress updates for each major step
3. WHEN any step fails THEN the system SHALL halt execution and report the failure point
4. WHEN the pipeline completes THEN the system SHALL display summary statistics for data and model performance
5. WHEN models are trained THEN the system SHALL automatically update the .env file to enable AI models

### Requirement 10

**User Story:** As a quality assurance engineer, I want to compare model performance on synthetic vs real data, so that I can quantify the improvement from using actual imagery.

#### Acceptance Criteria

1. WHEN comparing models THEN the system SHALL evaluate both synthetic-trained and real-trained models on the same test set
2. WHEN measuring performance THEN the system SHALL report accuracy, precision, recall, and F1 scores for both models
3. WHEN analyzing results THEN the system SHALL identify which crop health classes improved most with real data
4. WHEN generating reports THEN the system SHALL create visualizations comparing confusion matrices
5. WHEN documenting findings THEN the system SHALL save comparison metrics to JSON file for future reference
