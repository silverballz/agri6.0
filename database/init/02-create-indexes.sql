-- Create optimized indexes for Agricultural Monitoring Platform
-- These indexes are designed for production performance

-- Satellite Images Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_satellite_images_acquisition_date 
    ON satellite_images (acquisition_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_satellite_images_tile_id 
    ON satellite_images (tile_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_satellite_images_cloud_coverage 
    ON satellite_images (cloud_coverage) WHERE cloud_coverage < 20;

-- Spatial index for satellite images
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_satellite_images_geometry 
    ON satellite_images USING GIST (geometry);

-- Composite index for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_satellite_images_tile_date 
    ON satellite_images (tile_id, acquisition_date DESC);

-- Monitoring Zones Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_monitoring_zones_name 
    ON monitoring_zones (name);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_monitoring_zones_crop_type 
    ON monitoring_zones (crop_type);

-- Spatial index for monitoring zones
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_monitoring_zones_geometry 
    ON monitoring_zones USING GIST (geometry);

-- Date range index for active zones
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_monitoring_zones_active_period 
    ON monitoring_zones (planting_date, expected_harvest) 
    WHERE planting_date IS NOT NULL AND expected_harvest IS NOT NULL;

-- Sensor Locations Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_locations_zone_id 
    ON sensor_locations (zone_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_locations_type 
    ON sensor_locations (sensor_type);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_locations_status 
    ON sensor_locations (status) WHERE status = 'active';

-- Spatial index for sensor locations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_locations_location 
    ON sensor_locations USING GIST (location);

-- Index Timeseries Indexes (Critical for performance)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_index_timeseries_zone_timestamp 
    ON index_timeseries (zone_id, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_index_timeseries_type_timestamp 
    ON index_timeseries (index_type, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_index_timeseries_image_id 
    ON index_timeseries (image_id);

-- Composite index for trend queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_index_timeseries_zone_type_timestamp 
    ON index_timeseries (zone_id, index_type, timestamp DESC);

-- Quality-filtered index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_index_timeseries_quality 
    ON index_timeseries (quality_score) WHERE quality_score >= 0.7;

-- Sensor Readings Indexes (High volume table)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_readings_sensor_timestamp 
    ON sensor_readings (sensor_id, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_readings_timestamp 
    ON sensor_readings (timestamp DESC);

-- Partial index for good quality readings
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_readings_quality_good 
    ON sensor_readings (sensor_id, timestamp DESC) 
    WHERE quality_flag = 'good';

-- Alerts Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_zone_id 
    ON alerts (zone_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_triggered_at 
    ON alerts (triggered_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_severity 
    ON alerts (severity);

-- Active alerts index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_active 
    ON alerts (zone_id, triggered_at DESC) 
    WHERE resolved_at IS NULL;

-- Composite index for alert dashboard
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_zone_severity_triggered 
    ON alerts (zone_id, severity, triggered_at DESC);

-- Model Performance Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_performance_name_version 
    ON model_performance (model_name, model_version);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_performance_evaluation_date 
    ON model_performance (evaluation_date DESC);

-- Composite index for model monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_performance_name_metric_date 
    ON model_performance (model_name, metric_name, evaluation_date DESC);

-- Create materialized views for common aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS zone_latest_indices AS
SELECT DISTINCT ON (zone_id, index_type) 
    zone_id,
    index_type,
    timestamp,
    mean_value,
    quality_score
FROM index_timeseries 
WHERE quality_score >= 0.7
ORDER BY zone_id, index_type, timestamp DESC;

-- Index for the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_zone_latest_indices_zone_type 
    ON zone_latest_indices (zone_id, index_type);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_zone_latest_indices()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY zone_latest_indices;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers to relevant tables
CREATE TRIGGER update_satellite_images_updated_at 
    BEFORE UPDATE ON satellite_images 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_monitoring_zones_updated_at 
    BEFORE UPDATE ON monitoring_zones 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();