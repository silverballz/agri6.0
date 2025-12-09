"""
Alert Notification Preferences Module

Manages user preferences for alert notifications including:
- Alert thresholds
- Alert type filtering
- Notification grouping
- Snooze functionality
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AlertPreferences:
    """
    User preferences for alert notifications.
    
    Attributes:
        enabled_alert_types: Set of enabled alert types
        severity_threshold: Minimum severity level to show ('low', 'medium', 'high', 'critical')
        ndvi_threshold: Custom NDVI threshold for vegetation stress alerts
        ndwi_threshold: Custom NDWI threshold for water stress alerts
        group_similar_alerts: Whether to group similar alerts together
        snooze_duration_hours: Default snooze duration in hours
        snoozed_alerts: Dictionary mapping alert IDs to snooze expiry times
        email_notifications: Enable email notifications
        sms_notifications: Enable SMS notifications
        push_notifications: Enable push notifications
    """
    enabled_alert_types: Set[str] = None
    severity_threshold: str = 'low'
    ndvi_threshold: float = 0.4
    ndwi_threshold: float = -0.1
    group_similar_alerts: bool = True
    snooze_duration_hours: int = 24
    snoozed_alerts: Dict[int, str] = None
    email_notifications: bool = True
    sms_notifications: bool = False
    push_notifications: bool = True
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.enabled_alert_types is None:
            self.enabled_alert_types = {
                'vegetation_stress',
                'pest_risk',
                'disease_risk',
                'water_stress',
                'environmental'
            }
        if self.snoozed_alerts is None:
            self.snoozed_alerts = {}
    
    def to_dict(self) -> Dict:
        """Convert preferences to dictionary."""
        return {
            'enabled_alert_types': list(self.enabled_alert_types),
            'severity_threshold': self.severity_threshold,
            'ndvi_threshold': self.ndvi_threshold,
            'ndwi_threshold': self.ndwi_threshold,
            'group_similar_alerts': self.group_similar_alerts,
            'snooze_duration_hours': self.snooze_duration_hours,
            'snoozed_alerts': self.snoozed_alerts,
            'email_notifications': self.email_notifications,
            'sms_notifications': self.sms_notifications,
            'push_notifications': self.push_notifications
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AlertPreferences':
        """Create preferences from dictionary."""
        # Convert list back to set for enabled_alert_types
        if 'enabled_alert_types' in data:
            data['enabled_alert_types'] = set(data['enabled_alert_types'])
        return cls(**data)


class AlertPreferencesManager:
    """
    Manages loading, saving, and applying alert preferences.
    """
    
    SEVERITY_LEVELS = ['low', 'medium', 'high', 'critical']
    SEVERITY_ORDER = {level: i for i, level in enumerate(SEVERITY_LEVELS)}
    
    def __init__(self, preferences_file: str = "data/alert_preferences.json"):
        """
        Initialize preferences manager.
        
        Args:
            preferences_file: Path to preferences JSON file
        """
        self.preferences_file = Path(preferences_file)
        self.preferences = self.load_preferences()
    
    def load_preferences(self) -> AlertPreferences:
        """
        Load preferences from file or create default.
        
        Returns:
            AlertPreferences object
        """
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded alert preferences from {self.preferences_file}")
                    return AlertPreferences.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading preferences: {e}")
                return AlertPreferences()
        else:
            logger.info("No preferences file found, using defaults")
            return AlertPreferences()
    
    def save_preferences(self) -> bool:
        """
        Save current preferences to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.preferences_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences.to_dict(), f, indent=2)
            
            logger.info(f"Saved alert preferences to {self.preferences_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
            return False
    
    def should_show_alert(self, alert_type: str, severity: str) -> bool:
        """
        Check if an alert should be shown based on preferences.
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            
        Returns:
            True if alert should be shown, False otherwise
        """
        # Check if alert type is enabled
        if alert_type not in self.preferences.enabled_alert_types:
            return False
        
        # Check severity threshold
        alert_severity_level = self.SEVERITY_ORDER.get(severity, 0)
        threshold_level = self.SEVERITY_ORDER.get(self.preferences.severity_threshold, 0)
        
        return alert_severity_level >= threshold_level
    
    def snooze_alert(self, alert_id: int, hours: Optional[int] = None) -> datetime:
        """
        Snooze an alert for a specified duration.
        
        Args:
            alert_id: Alert ID to snooze
            hours: Duration in hours (uses default if not specified)
            
        Returns:
            Expiry datetime when snooze ends
        """
        duration = hours if hours is not None else self.preferences.snooze_duration_hours
        expiry = datetime.now() + timedelta(hours=duration)
        
        self.preferences.snoozed_alerts[alert_id] = expiry.isoformat()
        self.save_preferences()
        
        logger.info(f"Snoozed alert {alert_id} until {expiry}")
        return expiry
    
    def unsnooze_alert(self, alert_id: int) -> bool:
        """
        Remove snooze from an alert.
        
        Args:
            alert_id: Alert ID to unsnooze
            
        Returns:
            True if alert was snoozed, False otherwise
        """
        if alert_id in self.preferences.snoozed_alerts:
            del self.preferences.snoozed_alerts[alert_id]
            self.save_preferences()
            logger.info(f"Unsnoozed alert {alert_id}")
            return True
        return False
    
    def is_alert_snoozed(self, alert_id: int) -> bool:
        """
        Check if an alert is currently snoozed.
        
        Args:
            alert_id: Alert ID to check
            
        Returns:
            True if alert is snoozed, False otherwise
        """
        if alert_id not in self.preferences.snoozed_alerts:
            return False
        
        # Check if snooze has expired
        expiry_str = self.preferences.snoozed_alerts[alert_id]
        expiry = datetime.fromisoformat(expiry_str)
        
        if datetime.now() > expiry:
            # Snooze expired, remove it
            del self.preferences.snoozed_alerts[alert_id]
            self.save_preferences()
            return False
        
        return True
    
    def get_snooze_expiry(self, alert_id: int) -> Optional[datetime]:
        """
        Get the snooze expiry time for an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Expiry datetime or None if not snoozed
        """
        if alert_id in self.preferences.snoozed_alerts:
            return datetime.fromisoformat(self.preferences.snoozed_alerts[alert_id])
        return None
    
    def clean_expired_snoozes(self):
        """Remove all expired snoozes from preferences."""
        now = datetime.now()
        expired = []
        
        for alert_id, expiry_str in self.preferences.snoozed_alerts.items():
            expiry = datetime.fromisoformat(expiry_str)
            if now > expiry:
                expired.append(alert_id)
        
        for alert_id in expired:
            del self.preferences.snoozed_alerts[alert_id]
        
        if expired:
            self.save_preferences()
            logger.info(f"Cleaned {len(expired)} expired snoozes")
    
    def update_alert_type_filter(self, alert_types: Set[str]):
        """
        Update which alert types are enabled.
        
        Args:
            alert_types: Set of enabled alert type names
        """
        self.preferences.enabled_alert_types = alert_types
        self.save_preferences()
    
    def update_severity_threshold(self, threshold: str):
        """
        Update minimum severity threshold.
        
        Args:
            threshold: Severity level ('low', 'medium', 'high', 'critical')
        """
        if threshold in self.SEVERITY_LEVELS:
            self.preferences.severity_threshold = threshold
            self.save_preferences()
        else:
            logger.warning(f"Invalid severity threshold: {threshold}")
    
    def update_custom_thresholds(self, ndvi: Optional[float] = None, ndwi: Optional[float] = None):
        """
        Update custom index thresholds.
        
        Args:
            ndvi: NDVI threshold (optional)
            ndwi: NDWI threshold (optional)
        """
        if ndvi is not None:
            self.preferences.ndvi_threshold = ndvi
        if ndwi is not None:
            self.preferences.ndwi_threshold = ndwi
        self.save_preferences()
    
    def filter_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """
        Filter alerts based on preferences.
        
        Args:
            alerts: List of alert dictionaries
            
        Returns:
            Filtered list of alerts
        """
        filtered = []
        
        for alert in alerts:
            alert_id = alert.get('id')
            alert_type = alert.get('alert_type')
            severity = alert.get('severity')
            
            # Skip snoozed alerts
            if self.is_alert_snoozed(alert_id):
                continue
            
            # Check if alert should be shown
            if self.should_show_alert(alert_type, severity):
                filtered.append(alert)
        
        return filtered
    
    def group_alerts(self, alerts: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group similar alerts together if grouping is enabled.
        
        Args:
            alerts: List of alert dictionaries
            
        Returns:
            Dictionary mapping group keys to alert lists
        """
        if not self.preferences.group_similar_alerts:
            return {'all': alerts}
        
        groups = {}
        
        for alert in alerts:
            # Group by alert type and severity
            key = f"{alert.get('alert_type')}_{alert.get('severity')}"
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append(alert)
        
        return groups
