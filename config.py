"""
Configuration management for AgriFlux Dashboard.

This module provides centralized configuration management with support for
different environments (development, staging, production) and environment
variable overrides.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    @classmethod
    def from_env(cls):
        """Create database config from environment variables."""
        return cls(
            path=os.getenv('DATABASE_PATH', 'data/agriflux.db'),
            backup_enabled=os.getenv('DB_BACKUP_ENABLED', 'true').lower() == 'true',
            backup_interval_hours=int(os.getenv('DB_BACKUP_INTERVAL', '24'))
        )


@dataclass
class PathConfig:
    """File path configuration settings."""
    processed_data: Path
    sentinel_data: Path
    demo_data: Path
    model_path: Path
    log_path: Path
    
    @classmethod
    def from_env(cls):
        """Create path config from environment variables."""
        return cls(
            processed_data=Path(os.getenv('PROCESSED_DATA_PATH', 'data/processed/')),
            sentinel_data=Path(os.getenv('SENTINEL_DATA_PATH', 
                'S2A_MSIL2A_20240923T053641_N0511_R005_T43REQ_20240923T084448.SAFE')),
            demo_data=Path(os.getenv('DEMO_DATA_PATH', 'data/demo/')),
            model_path=Path(os.getenv('MODEL_PATH', 'models/')),
            log_path=Path(os.getenv('LOG_PATH', 'logs/'))
        )


@dataclass
class ModelConfig:
    """AI model configuration settings."""
    use_ai_models: bool
    fallback_to_rules: bool
    model_path: Path
    retrain_interval_days: int = 7
    performance_threshold: float = 0.85
    
    @classmethod
    def from_env(cls):
        """Create model config from environment variables."""
        return cls(
            use_ai_models=os.getenv('USE_AI_MODELS', 'false').lower() == 'true',
            fallback_to_rules=os.getenv('FALLBACK_TO_RULES', 'true').lower() == 'true',
            model_path=Path(os.getenv('MODEL_PATH', 'models/')),
            retrain_interval_days=int(os.getenv('MODEL_RETRAIN_INTERVAL', '7')),
            performance_threshold=float(os.getenv('MODEL_PERFORMANCE_THRESHOLD', '0.85'))
        )


@dataclass
class DashboardConfig:
    """Dashboard UI configuration settings."""
    port: int
    title: str
    enable_demo_mode: bool
    theme: str = 'light'
    
    @classmethod
    def from_env(cls):
        """Create dashboard config from environment variables."""
        return cls(
            port=int(os.getenv('DASHBOARD_PORT', '8501')),
            title=os.getenv('DASHBOARD_TITLE', 'AgriFlux - Agricultural Monitoring Platform'),
            enable_demo_mode=os.getenv('ENABLE_DEMO_MODE', 'true').lower() == 'true',
            theme=os.getenv('DASHBOARD_THEME', 'light')
        )


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    log_path: Path
    log_file: str
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    @classmethod
    def from_env(cls):
        """Create logging config from environment variables."""
        return cls(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            log_path=Path(os.getenv('LOG_PATH', 'logs/')),
            log_file=os.getenv('LOG_FILE', 'dashboard.log'),
            max_file_size_mb=int(os.getenv('LOG_MAX_SIZE_MB', '10')),
            backup_count=int(os.getenv('LOG_BACKUP_COUNT', '5'))
        )


@dataclass
class AlertConfig:
    """Alert system configuration settings."""
    retention_days: int
    max_active_alerts: int
    enable_notifications: bool = False
    
    @classmethod
    def from_env(cls):
        """Create alert config from environment variables."""
        return cls(
            retention_days=int(os.getenv('ALERT_RETENTION_DAYS', '30')),
            max_active_alerts=int(os.getenv('MAX_ACTIVE_ALERTS', '100')),
            enable_notifications=os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'
        )


@dataclass
class PerformanceConfig:
    """Performance and optimization configuration settings."""
    cache_enabled: bool
    cache_ttl: int
    max_workers: int
    batch_size: int = 1000
    memory_limit_gb: float = 8.0
    
    @classmethod
    def from_env(cls):
        """Create performance config from environment variables."""
        return cls(
            cache_enabled=os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            cache_ttl=int(os.getenv('CACHE_TTL', '3600')),
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            batch_size=int(os.getenv('BATCH_SIZE', '1000')),
            memory_limit_gb=float(os.getenv('MEMORY_LIMIT_GB', '8.0'))
        )


@dataclass
class FeatureFlags:
    """Feature flag configuration."""
    enable_export: bool
    enable_reports: bool
    enable_temporal_analysis: bool
    enable_usp_features: bool
    
    @classmethod
    def from_env(cls):
        """Create feature flags from environment variables."""
        return cls(
            enable_export=os.getenv('ENABLE_EXPORT', 'true').lower() == 'true',
            enable_reports=os.getenv('ENABLE_REPORTS', 'true').lower() == 'true',
            enable_temporal_analysis=os.getenv('ENABLE_TEMPORAL_ANALYSIS', 'true').lower() == 'true',
            enable_usp_features=os.getenv('ENABLE_USP_FEATURES', 'true').lower() == 'true'
        )


class Config:
    """
    Main configuration class that aggregates all configuration sections.
    
    Usage:
        from config import config
        
        # Access configuration
        db_path = config.database.path
        log_level = config.logging.level
        
        # Check environment
        if config.is_production():
            # Production-specific logic
            pass
    """
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            environment: Environment name (development, staging, production).
                        If None, reads from AGRIFLUX_ENV environment variable.
        """
        self.environment = environment or os.getenv('AGRIFLUX_ENV', 'development')
        
        # Load all configuration sections
        self.database = DatabaseConfig.from_env()
        self.paths = PathConfig.from_env()
        self.models = ModelConfig.from_env()
        self.dashboard = DashboardConfig.from_env()
        self.logging = LoggingConfig.from_env()
        self.alerts = AlertConfig.from_env()
        self.performance = PerformanceConfig.from_env()
        self.features = FeatureFlags.from_env()
        
        # Ensure required directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            self.paths.processed_data,
            self.paths.demo_data,
            self.paths.model_path,
            self.paths.log_path,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == 'development'
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment.lower() == 'staging'
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return f"sqlite:///{self.database.path}"
    
    def get_log_file_path(self) -> Path:
        """Get full path to log file."""
        return self.paths.log_path / self.logging.log_file
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment,
            'database': {
                'path': self.database.path,
                'backup_enabled': self.database.backup_enabled,
            },
            'paths': {
                'processed_data': str(self.paths.processed_data),
                'sentinel_data': str(self.paths.sentinel_data),
                'demo_data': str(self.paths.demo_data),
                'model_path': str(self.paths.model_path),
                'log_path': str(self.paths.log_path),
            },
            'models': {
                'use_ai_models': self.models.use_ai_models,
                'fallback_to_rules': self.models.fallback_to_rules,
            },
            'dashboard': {
                'port': self.dashboard.port,
                'title': self.dashboard.title,
                'enable_demo_mode': self.dashboard.enable_demo_mode,
            },
            'logging': {
                'level': self.logging.level,
                'log_file': self.logging.log_file,
            },
            'features': {
                'enable_export': self.features.enable_export,
                'enable_reports': self.features.enable_reports,
                'enable_temporal_analysis': self.features.enable_temporal_analysis,
                'enable_usp_features': self.features.enable_usp_features,
            }
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(environment='{self.environment}')"


# Global configuration instance
config = Config()


# Environment-specific configuration presets
class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    def __init__(self):
        super().__init__('development')
        # Development-specific overrides
        self.logging.level = 'DEBUG'
        self.dashboard.enable_demo_mode = True
        self.models.use_ai_models = False


class StagingConfig(Config):
    """Staging environment configuration."""
    
    def __init__(self):
        super().__init__('staging')
        # Staging-specific overrides
        self.logging.level = 'INFO'
        self.dashboard.enable_demo_mode = True
        self.models.use_ai_models = True


class ProductionConfig(Config):
    """Production environment configuration."""
    
    def __init__(self):
        super().__init__('production')
        # Production-specific overrides
        self.logging.level = 'WARNING'
        self.dashboard.enable_demo_mode = False
        self.models.use_ai_models = True
        self.database.backup_enabled = True


def get_config(environment: Optional[str] = None) -> Config:
    """
    Get configuration for specified environment.
    
    Args:
        environment: Environment name (development, staging, production).
                    If None, uses AGRIFLUX_ENV environment variable.
    
    Returns:
        Configuration instance for the specified environment.
    """
    env = environment or os.getenv('AGRIFLUX_ENV', 'development')
    
    if env.lower() == 'development':
        return DevelopmentConfig()
    elif env.lower() == 'staging':
        return StagingConfig()
    elif env.lower() == 'production':
        return ProductionConfig()
    else:
        return Config(env)


if __name__ == '__main__':
    # Print current configuration
    print("Current Configuration:")
    print("=" * 60)
    print(f"Environment: {config.environment}")
    print(f"Database: {config.database.path}")
    print(f"Log Level: {config.logging.level}")
    print(f"Dashboard Port: {config.dashboard.port}")
    print(f"Demo Mode: {config.dashboard.enable_demo_mode}")
    print(f"AI Models: {config.models.use_ai_models}")
    print("=" * 60)
    
    # Print full configuration as dict
    import json
    print("\nFull Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
