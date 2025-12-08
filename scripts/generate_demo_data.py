"""
Generate demo data for AgriFlux Dashboard

Creates 3 field scenarios (healthy, stressed, mixed) with 5 time points each,
sample alerts for each severity level, and sample AI predictions.
Saves data as pickle files in data/demo/ directory.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoDataGenerator:
    """Generate realistic demo data for dashboard demonstrations."""
    
    def __init__(self, output_dir: str = "data/demo"):
        """
        Initialize demo data generator.
        
        Args:
            output_dir: Directory to save demo data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Image dimensions for demo data
        self.img_height = 100
        self.img_width = 100
        
        # Base date for time series
        self.base_date = datetime(2024, 9, 1)
        
        logger.info(f"Initialized DemoDataGenerator, output: {self.output_dir}")
    
    def generate_all_demo_data(self):
        """Generate all demo data: scenarios, time series, alerts, predictions."""
        logger.info("Starting demo data generation...")
        
        # Generate 3 field scenarios
        scenarios = self._generate_field_scenarios()
        
        # Generate time series for each scenario (5 time points)
        time_series = self._generate_time_series(scenarios)
        
        # Generate sample alerts
        alerts = self._generate_sample_alerts()
        
        # Generate sample predictions
        predictions = self._generate_sample_predictions(scenarios)
        
        # Save all data
        self._save_demo_data(scenarios, time_series, alerts, predictions)
        
        logger.info("Demo data generation complete!")
    
    def _generate_field_scenarios(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate 3 field scenarios with different health conditions.
        
        Returns:
            Dictionary with scenario names and their vegetation indices
        """
        logger.info("Generating field scenarios...")
        
        scenarios = {}
        
        # Scenario 1: Healthy Field
        scenarios['healthy_field'] = self._create_healthy_field()
        
        # Scenario 2: Stressed Field
        scenarios['stressed_field'] = self._create_stressed_field()
        
        # Scenario 3: Mixed Field
        scenarios['mixed_field'] = self._create_mixed_field()
        
        logger.info(f"Generated {len(scenarios)} field scenarios")
        return scenarios
    
    def _create_healthy_field(self) -> Dict[str, np.ndarray]:
        """Create a healthy field with high vegetation indices."""
        # Healthy NDVI: 0.7-0.9
        ndvi = np.random.uniform(0.7, 0.9, (self.img_height, self.img_width))
        
        # Add some natural variation
        ndvi = self._add_spatial_variation(ndvi, variation=0.05)
        
        # Calculate other indices based on NDVI
        savi = ndvi * 0.9  # SAVI slightly lower than NDVI
        evi = ndvi * 0.85  # EVI slightly lower
        ndwi = np.random.uniform(0.2, 0.4, (self.img_height, self.img_width))
        ndsi = np.random.uniform(-0.5, -0.3, (self.img_height, self.img_width))
        
        return {
            'ndvi': ndvi,
            'savi': savi,
            'evi': evi,
            'ndwi': ndwi,
            'ndsi': ndsi,
            'description': 'Healthy field with vigorous vegetation growth',
            'health_status': 'healthy'
        }
    
    def _create_stressed_field(self) -> Dict[str, np.ndarray]:
        """Create a stressed field with low vegetation indices."""
        # Stressed NDVI: 0.2-0.4
        ndvi = np.random.uniform(0.2, 0.4, (self.img_height, self.img_width))
        
        # Add stress patterns (patches of severe stress)
        ndvi = self._add_stress_patches(ndvi, num_patches=3, severity=0.15)
        
        # Calculate other indices
        savi = ndvi * 0.85
        evi = ndvi * 0.8
        ndwi = np.random.uniform(-0.3, -0.1, (self.img_height, self.img_width))
        ndsi = np.random.uniform(-0.4, -0.2, (self.img_height, self.img_width))
        
        return {
            'ndvi': ndvi,
            'savi': savi,
            'evi': evi,
            'ndwi': ndwi,
            'ndsi': ndsi,
            'description': 'Stressed field showing signs of water deficit and poor health',
            'health_status': 'stressed'
        }
    
    def _create_mixed_field(self) -> Dict[str, np.ndarray]:
        """Create a mixed field with varying health conditions."""
        # Create zones with different health levels
        ndvi = np.zeros((self.img_height, self.img_width))
        
        # Healthy zone (left third)
        ndvi[:, :self.img_width//3] = np.random.uniform(0.7, 0.85, 
                                                        (self.img_height, self.img_width//3))
        
        # Moderate zone (middle third)
        ndvi[:, self.img_width//3:2*self.img_width//3] = np.random.uniform(0.5, 0.65,
                                                                            (self.img_height, self.img_width//3))
        
        # Stressed zone (right third)
        ndvi[:, 2*self.img_width//3:] = np.random.uniform(0.25, 0.45,
                                                          (self.img_height, self.img_width - 2*self.img_width//3))
        
        # Smooth transitions
        ndvi = self._smooth_transitions(ndvi)
        
        # Calculate other indices
        savi = ndvi * 0.88
        evi = ndvi * 0.83
        ndwi = np.random.uniform(-0.1, 0.3, (self.img_height, self.img_width))
        ndsi = np.random.uniform(-0.5, -0.2, (self.img_height, self.img_width))
        
        return {
            'ndvi': ndvi,
            'savi': savi,
            'evi': evi,
            'ndwi': ndwi,
            'ndsi': ndsi,
            'description': 'Mixed field with varying health conditions across zones',
            'health_status': 'mixed'
        }
    
    def _add_spatial_variation(self, data: np.ndarray, variation: float) -> np.ndarray:
        """Add natural spatial variation to data."""
        noise = np.random.normal(0, variation, data.shape)
        result = data + noise
        return np.clip(result, -1, 1)
    
    def _add_stress_patches(self, data: np.ndarray, num_patches: int, 
                           severity: float) -> np.ndarray:
        """Add patches of severe stress to the field."""
        result = data.copy()
        
        for _ in range(num_patches):
            # Random patch location and size
            center_y = np.random.randint(20, self.img_height - 20)
            center_x = np.random.randint(20, self.img_width - 20)
            radius = np.random.randint(10, 20)
            
            # Create circular patch
            y, x = np.ogrid[:self.img_height, :self.img_width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Apply stress (reduce NDVI)
            result[mask] = np.minimum(result[mask], severity)
        
        return result
    
    def _smooth_transitions(self, data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Smooth transitions between zones using simple averaging."""
        from scipy.ndimage import uniform_filter
        return uniform_filter(data, size=kernel_size)
    
    def _generate_time_series(self, scenarios: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        Generate 5 time points for each scenario showing temporal evolution.
        
        Args:
            scenarios: Dictionary of field scenarios
        
        Returns:
            Dictionary mapping scenario names to time series data
        """
        logger.info("Generating time series data...")
        
        time_series = {}
        
        for scenario_name, scenario_data in scenarios.items():
            series = []
            
            for i in range(5):
                # Date for this time point (every 16 days - Sentinel-2 revisit)
                date = self.base_date + timedelta(days=i * 16)
                
                # Create temporal variation based on scenario
                if scenario_data['health_status'] == 'healthy':
                    # Healthy field: slight improvement over time
                    trend_factor = 1.0 + (i * 0.02)
                elif scenario_data['health_status'] == 'stressed':
                    # Stressed field: gradual decline
                    trend_factor = 1.0 - (i * 0.03)
                else:  # mixed
                    # Mixed field: some recovery in middle period
                    trend_factor = 1.0 + (0.02 if i == 2 or i == 3 else 0)
                
                # Apply trend to indices
                time_point = {
                    'date': date.isoformat(),
                    'acquisition_date': date.strftime('%Y-%m-%d'),
                    'tile_id': f'DEMO_{scenario_name.upper()}',
                    'cloud_coverage': np.random.uniform(0, 15),
                    'ndvi': np.clip(scenario_data['ndvi'] * trend_factor, -1, 1),
                    'savi': np.clip(scenario_data['savi'] * trend_factor, -1, 1),
                    'evi': np.clip(scenario_data['evi'] * trend_factor, -1, 1),
                    'ndwi': scenario_data['ndwi'],
                    'ndsi': scenario_data['ndsi'],
                    'metadata': {
                        'scenario': scenario_name,
                        'time_point': i + 1,
                        'description': scenario_data['description']
                    }
                }
                
                # Calculate mean values for easy access
                time_point['mean_ndvi'] = float(np.mean(time_point['ndvi']))
                time_point['mean_savi'] = float(np.mean(time_point['savi']))
                time_point['mean_evi'] = float(np.mean(time_point['evi']))
                time_point['mean_ndwi'] = float(np.mean(time_point['ndwi']))
                
                series.append(time_point)
            
            time_series[scenario_name] = series
        
        logger.info(f"Generated time series with 5 points for {len(scenarios)} scenarios")
        return time_series
    
    def _generate_sample_alerts(self) -> List[Dict[str, Any]]:
        """Generate sample alerts for each severity level."""
        logger.info("Generating sample alerts...")
        
        alerts = [
            # Critical alerts
            {
                'id': 1,
                'alert_type': 'vegetation_stress',
                'severity': 'critical',
                'message': 'Severe vegetation stress detected: 45.2% of area has NDVI ≤ 0.3',
                'recommendation': 'IMMEDIATE ACTION REQUIRED: Inspect affected areas for pest damage, disease, or irrigation failure. Consider emergency irrigation and soil testing.',
                'affected_area_percentage': 45.2,
                'created_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'acknowledged': False,
                'scenario': 'stressed_field'
            },
            {
                'id': 2,
                'alert_type': 'water_stress',
                'severity': 'critical',
                'message': 'Severe water stress: 38.7% of area has NDWI ≤ -0.2',
                'recommendation': 'IMMEDIATE IRRIGATION REQUIRED. Check irrigation system functionality and increase watering frequency.',
                'affected_area_percentage': 38.7,
                'created_at': (datetime.now() - timedelta(hours=3)).isoformat(),
                'acknowledged': False,
                'scenario': 'stressed_field'
            },
            
            # High severity alerts
            {
                'id': 3,
                'alert_type': 'disease_risk',
                'severity': 'high',
                'message': 'High fungal disease risk: Humidity 82.5% and temperature 26.3°C favor fungal growth',
                'recommendation': 'Inspect crops for early signs of fungal infection (leaf spots, mildew). Consider preventive fungicide application and improve air circulation.',
                'affected_area_percentage': 100.0,
                'created_at': (datetime.now() - timedelta(hours=5)).isoformat(),
                'acknowledged': False,
                'scenario': 'mixed_field'
            },
            {
                'id': 4,
                'alert_type': 'vegetation_stress',
                'severity': 'high',
                'message': 'High vegetation stress detected: 22.3% of area has NDVI between 0.3 and 0.4',
                'recommendation': 'Increase irrigation frequency, monitor closely for pest activity, and consider nutrient supplementation.',
                'affected_area_percentage': 22.3,
                'created_at': (datetime.now() - timedelta(hours=8)).isoformat(),
                'acknowledged': True,
                'acknowledged_at': (datetime.now() - timedelta(hours=6)).isoformat(),
                'scenario': 'mixed_field'
            },
            
            # Medium severity alerts
            {
                'id': 5,
                'alert_type': 'pest_risk',
                'severity': 'medium',
                'message': 'Elevated pest activity risk: Temperature 29.1°C and humidity 65.2% favor insect reproduction',
                'recommendation': 'Increase pest monitoring frequency. Check for aphids, thrips, and other common pests. Consider integrated pest management strategies.',
                'affected_area_percentage': 100.0,
                'created_at': (datetime.now() - timedelta(hours=12)).isoformat(),
                'acknowledged': True,
                'acknowledged_at': (datetime.now() - timedelta(hours=10)).isoformat(),
                'scenario': 'healthy_field'
            },
            {
                'id': 6,
                'alert_type': 'vegetation_stress',
                'severity': 'medium',
                'message': 'Moderate vegetation stress: 15.8% of area has NDVI between 0.4 and 0.5',
                'recommendation': 'Review irrigation schedule, monitor weather conditions, and prepare for potential intervention.',
                'affected_area_percentage': 15.8,
                'created_at': (datetime.now() - timedelta(days=1)).isoformat(),
                'acknowledged': True,
                'acknowledged_at': (datetime.now() - timedelta(hours=20)).isoformat(),
                'scenario': 'mixed_field'
            },
            
            # Low severity alerts
            {
                'id': 7,
                'alert_type': 'environmental',
                'severity': 'low',
                'message': 'Low humidity alert: 38.2% may increase water stress',
                'recommendation': 'Increase irrigation frequency to compensate for high evapotranspiration rates.',
                'affected_area_percentage': 100.0,
                'created_at': (datetime.now() - timedelta(days=2)).isoformat(),
                'acknowledged': True,
                'acknowledged_at': (datetime.now() - timedelta(days=1, hours=20)).isoformat(),
                'scenario': 'healthy_field'
            },
            {
                'id': 8,
                'alert_type': 'vegetation_stress',
                'severity': 'low',
                'message': 'Minor vegetation stress: 12.4% of area has NDVI between 0.5 and 0.6',
                'recommendation': 'Continue routine monitoring. Consider optimizing irrigation and fertilization schedules.',
                'affected_area_percentage': 12.4,
                'created_at': (datetime.now() - timedelta(days=3)).isoformat(),
                'acknowledged': True,
                'acknowledged_at': (datetime.now() - timedelta(days=2, hours=18)).isoformat(),
                'scenario': 'healthy_field'
            }
        ]
        
        logger.info(f"Generated {len(alerts)} sample alerts")
        return alerts
    
    def _generate_sample_predictions(self, scenarios: Dict[str, Dict]) -> Dict[str, Dict]:
        """Generate sample AI predictions for each scenario."""
        logger.info("Generating sample predictions...")
        
        predictions = {}
        
        for scenario_name, scenario_data in scenarios.items():
            ndvi = scenario_data['ndvi']
            
            # Generate predictions based on NDVI (simulating AI model output)
            # Classes: 0=healthy, 1=moderate, 2=stressed, 3=critical
            pred_array = np.zeros_like(ndvi, dtype=int)
            pred_array[ndvi > 0.7] = 0  # healthy
            pred_array[(ndvi > 0.5) & (ndvi <= 0.7)] = 1  # moderate
            pred_array[(ndvi > 0.3) & (ndvi <= 0.5)] = 2  # stressed
            pred_array[ndvi <= 0.3] = 3  # critical
            
            # Generate confidence scores (higher for more extreme values)
            confidence = np.abs(ndvi - 0.5) / 0.5
            confidence = np.clip(confidence, 0.6, 0.95)  # Realistic confidence range
            
            predictions[scenario_name] = {
                'predictions': pred_array,
                'confidence_scores': confidence,
                'class_names': ['Healthy', 'Moderate', 'Stressed', 'Critical'],
                'model_version': 'demo_rule_based_v1.0',
                'prediction_type': 'crop_health',
                'created_at': datetime.now().isoformat(),
                'metadata': {
                    'scenario': scenario_name,
                    'method': 'rule_based',
                    'class_distribution': {
                        'healthy': int(np.sum(pred_array == 0)),
                        'moderate': int(np.sum(pred_array == 1)),
                        'stressed': int(np.sum(pred_array == 2)),
                        'critical': int(np.sum(pred_array == 3))
                    }
                }
            }
        
        logger.info(f"Generated predictions for {len(scenarios)} scenarios")
        return predictions
    
    def _save_demo_data(self, scenarios: Dict, time_series: Dict, 
                       alerts: List, predictions: Dict):
        """Save all demo data to pickle files."""
        logger.info("Saving demo data...")
        
        # Save scenarios
        scenarios_file = self.output_dir / 'scenarios.pkl'
        with open(scenarios_file, 'wb') as f:
            pickle.dump(scenarios, f)
        logger.info(f"Saved scenarios to {scenarios_file}")
        
        # Save time series
        time_series_file = self.output_dir / 'time_series.pkl'
        with open(time_series_file, 'wb') as f:
            pickle.dump(time_series, f)
        logger.info(f"Saved time series to {time_series_file}")
        
        # Save alerts
        alerts_file = self.output_dir / 'alerts.pkl'
        with open(alerts_file, 'wb') as f:
            pickle.dump(alerts, f)
        logger.info(f"Saved alerts to {alerts_file}")
        
        # Save predictions
        predictions_file = self.output_dir / 'predictions.pkl'
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions, f)
        logger.info(f"Saved predictions to {predictions_file}")
        
        # Also save a metadata file with summary info
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'num_scenarios': len(scenarios),
            'num_time_points': 5,
            'num_alerts': len(alerts),
            'image_dimensions': (self.img_height, self.img_width),
            'scenarios': list(scenarios.keys()),
            'alert_severities': {
                'critical': sum(1 for a in alerts if a['severity'] == 'critical'),
                'high': sum(1 for a in alerts if a['severity'] == 'high'),
                'medium': sum(1 for a in alerts if a['severity'] == 'medium'),
                'low': sum(1 for a in alerts if a['severity'] == 'low')
            }
        }
        
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")
        
        logger.info("All demo data saved successfully!")


def main():
    """Main function to generate demo data."""
    print("=" * 60)
    print("AgriFlux Demo Data Generator")
    print("=" * 60)
    
    generator = DemoDataGenerator()
    generator.generate_all_demo_data()
    
    print("\n" + "=" * 60)
    print("Demo data generation complete!")
    print(f"Files saved to: {generator.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
