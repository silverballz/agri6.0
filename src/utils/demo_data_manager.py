"""
Demo Data Manager for AgriFlux Dashboard

Manages loading and providing demo data for quick demonstrations.
Supports 3 field scenarios with time series, alerts, and predictions.
"""

import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DemoDataManager:
    """
    Manages demo data loading and access for dashboard demonstrations.
    
    Provides easy access to pre-generated demo data including:
    - 3 field scenarios (healthy, stressed, mixed)
    - 5 time points for each scenario
    - Sample alerts for all severity levels
    - Sample AI predictions
    """
    
    def __init__(self, demo_data_dir: str = "data/demo"):
        """
        Initialize demo data manager.
        
        Args:
            demo_data_dir: Directory containing demo data files
        """
        self.demo_data_dir = Path(demo_data_dir)
        self._scenarios = None
        self._time_series = None
        self._alerts = None
        self._predictions = None
        self._metadata = None
        self._is_loaded = False
        
        logger.info(f"Initialized DemoDataManager with dir: {self.demo_data_dir}")
    
    def is_demo_data_available(self) -> bool:
        """
        Check if demo data files exist.
        
        Returns:
            True if all required demo data files exist
        """
        required_files = [
            'scenarios.pkl',
            'time_series.pkl',
            'alerts.pkl',
            'predictions.pkl',
            'metadata.json'
        ]
        
        return all((self.demo_data_dir / f).exists() for f in required_files)
    
    def load_demo_data(self) -> bool:
        """
        Load all demo data from pickle files.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_demo_data_available():
            logger.error(f"Demo data files not found in {self.demo_data_dir}")
            logger.info("Run 'python scripts/generate_demo_data.py' to generate demo data")
            return False
        
        try:
            # Load scenarios
            with open(self.demo_data_dir / 'scenarios.pkl', 'rb') as f:
                self._scenarios = pickle.load(f)
            logger.info(f"Loaded {len(self._scenarios)} scenarios")
            
            # Load time series
            with open(self.demo_data_dir / 'time_series.pkl', 'rb') as f:
                self._time_series = pickle.load(f)
            logger.info(f"Loaded time series for {len(self._time_series)} scenarios")
            
            # Load alerts
            with open(self.demo_data_dir / 'alerts.pkl', 'rb') as f:
                self._alerts = pickle.load(f)
            logger.info(f"Loaded {len(self._alerts)} alerts")
            
            # Load predictions
            with open(self.demo_data_dir / 'predictions.pkl', 'rb') as f:
                self._predictions = pickle.load(f)
            logger.info(f"Loaded predictions for {len(self._predictions)} scenarios")
            
            # Load metadata
            with open(self.demo_data_dir / 'metadata.json', 'r') as f:
                self._metadata = json.load(f)
            logger.info("Loaded metadata")
            
            self._is_loaded = True
            logger.info("Demo data loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load demo data: {e}")
            self._is_loaded = False
            return False
    
    def get_scenario_names(self) -> List[str]:
        """
        Get list of available scenario names.
        
        Returns:
            List of scenario names
        """
        if not self._is_loaded:
            return []
        return list(self._scenarios.keys())
    
    def get_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific scenario.
        
        Args:
            scenario_name: Name of scenario ('healthy_field', 'stressed_field', 'mixed_field')
        
        Returns:
            Dictionary with scenario data or None if not found
        """
        if not self._is_loaded:
            logger.warning("Demo data not loaded. Call load_demo_data() first.")
            return None
        
        return self._scenarios.get(scenario_name)
    
    def get_time_series(self, scenario_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get time series data for a specific scenario.
        
        Args:
            scenario_name: Name of scenario
        
        Returns:
            List of time point dictionaries or None if not found
        """
        if not self._is_loaded:
            logger.warning("Demo data not loaded. Call load_demo_data() first.")
            return None
        
        return self._time_series.get(scenario_name)
    
    def get_time_point(self, scenario_name: str, time_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific time point for a scenario.
        
        Args:
            scenario_name: Name of scenario
            time_index: Index of time point (0-4)
        
        Returns:
            Time point dictionary or None if not found
        """
        time_series = self.get_time_series(scenario_name)
        if time_series is None or time_index >= len(time_series):
            return None
        
        return time_series[time_index]
    
    def get_latest_time_point(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent time point for a scenario.
        
        Args:
            scenario_name: Name of scenario
        
        Returns:
            Latest time point dictionary or None if not found
        """
        time_series = self.get_time_series(scenario_name)
        if time_series is None or len(time_series) == 0:
            return None
        
        return time_series[-1]  # Last time point
    
    def get_alerts(self, scenario_name: Optional[str] = None,
                  severity: Optional[str] = None,
                  acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get alerts with optional filtering.
        
        Args:
            scenario_name: Optional scenario filter
            severity: Optional severity filter ('critical', 'high', 'medium', 'low')
            acknowledged: Optional acknowledgment status filter
        
        Returns:
            List of alert dictionaries
        """
        if not self._is_loaded:
            logger.warning("Demo data not loaded. Call load_demo_data() first.")
            return []
        
        alerts = self._alerts.copy()
        
        # Apply filters
        if scenario_name is not None:
            alerts = [a for a in alerts if a.get('scenario') == scenario_name]
        
        if severity is not None:
            alerts = [a for a in alerts if a.get('severity') == severity]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.get('acknowledged') == acknowledged]
        
        return alerts
    
    def get_active_alerts(self, scenario_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get unacknowledged alerts.
        
        Args:
            scenario_name: Optional scenario filter
        
        Returns:
            List of active alert dictionaries
        """
        return self.get_alerts(scenario_name=scenario_name, acknowledged=False)
    
    def get_predictions(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Get AI predictions for a specific scenario.
        
        Args:
            scenario_name: Name of scenario
        
        Returns:
            Predictions dictionary or None if not found
        """
        if not self._is_loaded:
            logger.warning("Demo data not loaded. Call load_demo_data() first.")
            return None
        
        return self._predictions.get(scenario_name)
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get demo data metadata.
        
        Returns:
            Metadata dictionary or None if not loaded
        """
        return self._metadata
    
    def get_scenario_description(self, scenario_name: str) -> str:
        """
        Get human-readable description of a scenario.
        
        Args:
            scenario_name: Name of scenario
        
        Returns:
            Description string
        """
        descriptions = {
            'healthy_field': '🌱 Healthy Field - Vigorous vegetation growth with high NDVI values (0.7-0.9)',
            'stressed_field': '⚠️ Stressed Field - Significant vegetation stress with low NDVI values (0.2-0.4)',
            'mixed_field': '🔄 Mixed Field - Varying health conditions across different zones'
        }
        
        return descriptions.get(scenario_name, scenario_name)
    
    def get_summary_stats(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a scenario.
        
        Args:
            scenario_name: Name of scenario
        
        Returns:
            Dictionary with summary statistics
        """
        if not self._is_loaded:
            return {}
        
        scenario = self.get_scenario(scenario_name)
        time_series = self.get_time_series(scenario_name)
        alerts = self.get_alerts(scenario_name=scenario_name)
        predictions = self.get_predictions(scenario_name)
        
        if scenario is None:
            return {}
        
        import numpy as np
        
        stats = {
            'scenario_name': scenario_name,
            'description': scenario.get('description', ''),
            'health_status': scenario.get('health_status', 'unknown'),
            'current_ndvi': {
                'mean': float(np.mean(scenario['ndvi'])),
                'min': float(np.min(scenario['ndvi'])),
                'max': float(np.max(scenario['ndvi'])),
                'std': float(np.std(scenario['ndvi']))
            },
            'time_points': len(time_series) if time_series else 0,
            'total_alerts': len(alerts),
            'active_alerts': len([a for a in alerts if not a.get('acknowledged', False)]),
            'alert_breakdown': {}
        }
        
        # Alert breakdown by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            count = len([a for a in alerts if a.get('severity') == severity])
            if count > 0:
                stats['alert_breakdown'][severity] = count
        
        # Prediction stats
        if predictions:
            pred_array = predictions['predictions']
            class_dist = predictions['metadata'].get('class_distribution', {})
            stats['prediction_distribution'] = class_dist
        
        return stats
    
    def format_for_dashboard(self, scenario_name: str, 
                           time_index: int = -1) -> Dict[str, Any]:
        """
        Format demo data for dashboard consumption.
        
        Args:
            scenario_name: Name of scenario
            time_index: Time point index (-1 for latest)
        
        Returns:
            Dictionary formatted for dashboard use
        """
        if not self._is_loaded:
            return {}
        
        # Get time point data
        if time_index == -1:
            time_point = self.get_latest_time_point(scenario_name)
        else:
            time_point = self.get_time_point(scenario_name, time_index)
        
        if time_point is None:
            return {}
        
        # Get associated data
        alerts = self.get_active_alerts(scenario_name)
        predictions = self.get_predictions(scenario_name)
        
        # Format for dashboard
        dashboard_data = {
            'imagery': {
                'acquisition_date': time_point['acquisition_date'],
                'tile_id': time_point['tile_id'],
                'cloud_coverage': time_point['cloud_coverage'],
                'ndvi': time_point['ndvi'],
                'savi': time_point['savi'],
                'evi': time_point['evi'],
                'ndwi': time_point['ndwi'],
                'ndsi': time_point['ndsi'],
                'mean_ndvi': time_point['mean_ndvi'],
                'mean_savi': time_point['mean_savi'],
                'mean_evi': time_point['mean_evi'],
                'mean_ndwi': time_point['mean_ndwi'],
                'metadata': time_point['metadata']
            },
            'alerts': alerts,
            'predictions': predictions,
            'scenario_info': {
                'name': scenario_name,
                'description': self.get_scenario_description(scenario_name),
                'time_point': time_index if time_index >= 0 else len(self.get_time_series(scenario_name)) - 1
            }
        }
        
        return dashboard_data
    
    def is_loaded(self) -> bool:
        """
        Check if demo data is loaded.
        
        Returns:
            True if data is loaded
        """
        return self._is_loaded


# Singleton instance for easy access
_demo_manager_instance = None


def get_demo_manager(demo_data_dir: str = "data/demo") -> DemoDataManager:
    """
    Get singleton instance of DemoDataManager.
    
    Args:
        demo_data_dir: Directory containing demo data
    
    Returns:
        DemoDataManager instance
    """
    global _demo_manager_instance
    
    if _demo_manager_instance is None:
        _demo_manager_instance = DemoDataManager(demo_data_dir)
    
    return _demo_manager_instance
