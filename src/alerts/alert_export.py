"""
Alert Export and Reporting Module

Provides functionality to export alerts in various formats:
- CSV export with full details
- PDF summary reports
- Email notification templates
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import StringIO

logger = logging.getLogger(__name__)


class AlertExporter:
    """
    Handles exporting alerts to various formats.
    """
    
    def __init__(self):
        """Initialize alert exporter."""
        logger.info("Initialized AlertExporter")
    
    def export_to_csv(self, alerts: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """
        Export alerts to CSV format.
        
        Args:
            alerts: List of alert dictionaries
            include_metadata: Whether to include metadata column
            
        Returns:
            CSV string
        """
        if not alerts:
            return "No alerts to export"
        
        output = StringIO()
        
        # Define CSV columns
        fieldnames = [
            'id',
            'alert_type',
            'severity',
            'message',
            'recommendation',
            'affected_area_percentage',
            'field_name',
            'coordinates',
            'historical_context',
            'rate_of_change',
            'priority_score',
            'created_at',
            'acknowledged',
            'acknowledged_at'
        ]
        
        if include_metadata:
            fieldnames.append('metadata')
        
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for alert in alerts:
            # Parse metadata if it's a JSON string
            metadata = alert.get('metadata')
            if metadata and isinstance(metadata, str):
                try:
                    metadata_dict = json.loads(metadata)
                    # Extract enhanced fields from metadata
                    alert['field_name'] = metadata_dict.get('field_name', '')
                    alert['coordinates'] = str(metadata_dict.get('coordinates', ''))
                    alert['historical_context'] = metadata_dict.get('historical_context', '')
                    alert['rate_of_change'] = metadata_dict.get('rate_of_change', '')
                    alert['priority_score'] = metadata_dict.get('priority_score', 0)
                except:
                    pass
            
            # Calculate affected area percentage if not present
            if 'affected_area_percentage' not in alert:
                alert['affected_area_percentage'] = 0
            
            writer.writerow(alert)
        
        csv_content = output.getvalue()
        output.close()
        
        logger.info(f"Exported {len(alerts)} alerts to CSV")
        return csv_content
    
    def generate_summary_report(self, alerts: List[Dict[str, Any]]) -> str:
        """
        Generate a text summary report of alerts.
        
        Args:
            alerts: List of alert dictionaries
            
        Returns:
            Formatted text report
        """
        if not alerts:
            return "No alerts to report"
        
        report = []
        report.append("=" * 80)
        report.append("AGRIFLUX ALERT SUMMARY REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Alerts: {len(alerts)}")
        
        # Count by severity
        severity_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        report.append("\nAlerts by Severity:")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                report.append(f"  {severity.upper()}: {count}")
        
        # Count by type
        type_counts = {}
        for alert in alerts:
            alert_type = alert.get('alert_type', 'unknown')
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        report.append("\nAlerts by Type:")
        for alert_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            display_type = alert_type.replace('_', ' ').title()
            report.append(f"  {display_type}: {count}")
        
        # Acknowledgment status
        acknowledged = sum(1 for a in alerts if a.get('acknowledged', 0) == 1)
        pending = len(alerts) - acknowledged
        report.append(f"\nAcknowledgment Status:")
        report.append(f"  Acknowledged: {acknowledged}")
        report.append(f"  Pending: {pending}")
        
        report.append("")
        report.append("=" * 80)
        report.append("DETAILED ALERT LIST")
        report.append("=" * 80)
        report.append("")
        
        # Sort alerts by priority score if available, otherwise by severity
        sorted_alerts = sorted(
            alerts,
            key=lambda a: self._get_priority_score(a),
            reverse=True
        )
        
        # List top alerts
        for i, alert in enumerate(sorted_alerts[:20], 1):  # Show top 20
            report.append(f"Alert #{i} (ID: {alert.get('id', 'N/A')})")
            report.append(f"  Type: {alert.get('alert_type', 'unknown').replace('_', ' ').title()}")
            report.append(f"  Severity: {alert.get('severity', 'unknown').upper()}")
            
            # Add field name if available
            metadata = alert.get('metadata')
            if metadata:
                try:
                    metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                    field_name = metadata_dict.get('field_name')
                    if field_name:
                        report.append(f"  Location: {field_name}")
                    
                    priority_score = metadata_dict.get('priority_score')
                    if priority_score:
                        report.append(f"  Priority Score: {priority_score:.1f}/100")
                except:
                    pass
            
            report.append(f"  Message: {alert.get('message', 'No message')}")
            report.append(f"  Recommendation: {alert.get('recommendation', 'No recommendation')}")
            report.append(f"  Created: {alert.get('created_at', 'Unknown')}")
            report.append(f"  Status: {'Acknowledged' if alert.get('acknowledged', 0) == 1 else 'Pending'}")
            report.append("")
        
        if len(sorted_alerts) > 20:
            report.append(f"... and {len(sorted_alerts) - 20} more alerts")
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        logger.info(f"Generated summary report for {len(alerts)} alerts")
        return report_text
    
    def _get_priority_score(self, alert: Dict[str, Any]) -> float:
        """
        Extract priority score from alert, with fallback to severity-based score.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Priority score (0-100)
        """
        # Try to get priority score from metadata
        metadata = alert.get('metadata')
        if metadata:
            try:
                metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                priority_score = metadata_dict.get('priority_score')
                if priority_score is not None:
                    return float(priority_score)
            except:
                pass
        
        # Fallback to severity-based scoring
        severity = alert.get('severity', 'low')
        severity_scores = {
            'critical': 90,
            'high': 70,
            'medium': 50,
            'low': 30
        }
        return severity_scores.get(severity, 30)
    
    def generate_email_template(self, alerts: List[Dict[str, Any]], recipient_name: str = "User") -> str:
        """
        Generate an email notification template for alerts.
        
        Args:
            alerts: List of alert dictionaries
            recipient_name: Name of the recipient
            
        Returns:
            HTML email template
        """
        if not alerts:
            return "<p>No alerts to report.</p>"
        
        # Count critical and high severity alerts
        critical_count = sum(1 for a in alerts if a.get('severity') == 'critical')
        high_count = sum(1 for a in alerts if a.get('severity') == 'high')
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }")
        html.append(".header { background-color: #2e7d32; color: white; padding: 20px; text-align: center; }")
        html.append(".content { padding: 20px; }")
        html.append(".alert-box { border: 2px solid #ddd; border-radius: 8px; padding: 15px; margin: 15px 0; }")
        html.append(".critical { border-color: #d32f2f; background-color: #ffebee; }")
        html.append(".high { border-color: #f44336; background-color: #fff3e0; }")
        html.append(".medium { border-color: #ff9800; background-color: #fff8e1; }")
        html.append(".low { border-color: #4caf50; background-color: #e8f5e9; }")
        html.append(".alert-title { font-weight: bold; font-size: 18px; margin-bottom: 10px; }")
        html.append(".alert-message { margin: 10px 0; }")
        html.append(".alert-recommendation { background-color: rgba(255,255,255,0.7); padding: 10px; border-radius: 5px; margin-top: 10px; }")
        html.append(".footer { background-color: #f5f5f5; padding: 15px; text-align: center; font-size: 12px; color: #666; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append("<div class='header'>")
        html.append("<h1>🚨 AgriFlux Alert Notification</h1>")
        html.append("</div>")
        
        # Content
        html.append("<div class='content'>")
        html.append(f"<p>Dear {recipient_name},</p>")
        html.append(f"<p>You have <strong>{len(alerts)}</strong> active alert(s) requiring your attention:</p>")
        
        if critical_count > 0:
            html.append(f"<p style='color: #d32f2f; font-weight: bold;'>⚠️ {critical_count} CRITICAL alert(s) - Immediate action required!</p>")
        if high_count > 0:
            html.append(f"<p style='color: #f44336; font-weight: bold;'>⚠️ {high_count} HIGH priority alert(s)</p>")
        
        # Sort alerts by priority
        sorted_alerts = sorted(
            alerts,
            key=lambda a: self._get_priority_score(a),
            reverse=True
        )
        
        # Show top 10 alerts
        for alert in sorted_alerts[:10]:
            severity = alert.get('severity', 'medium')
            alert_type = alert.get('alert_type', 'unknown').replace('_', ' ').title()
            message = alert.get('message', 'No message')
            recommendation = alert.get('recommendation', 'No recommendation available')
            
            html.append(f"<div class='alert-box {severity}'>")
            html.append(f"<div class='alert-title'>{alert_type} - {severity.upper()}</div>")
            html.append(f"<div class='alert-message'>{message}</div>")
            html.append(f"<div class='alert-recommendation'><strong>💡 Recommendation:</strong> {recommendation}</div>")
            html.append("</div>")
        
        if len(sorted_alerts) > 10:
            html.append(f"<p><em>... and {len(sorted_alerts) - 10} more alert(s). Please log in to view all alerts.</em></p>")
        
        html.append("<p>Please log in to the AgriFlux dashboard to acknowledge these alerts and take appropriate action.</p>")
        html.append("<p>Best regards,<br>AgriFlux Alert System</p>")
        html.append("</div>")
        
        # Footer
        html.append("<div class='footer'>")
        html.append(f"<p>This is an automated notification from AgriFlux. Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append("<p>To manage your notification preferences, please visit the Alert Settings page in the dashboard.</p>")
        html.append("</div>")
        
        html.append("</body>")
        html.append("</html>")
        
        email_html = "\n".join(html)
        logger.info(f"Generated email template for {len(alerts)} alerts")
        return email_html
    
    def save_report_to_file(self, content: str, filename: str, output_dir: str = "data/exports") -> Path:
        """
        Save report content to a file.
        
        Args:
            content: Report content
            filename: Output filename
            output_dir: Output directory path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Saved report to {file_path}")
        return file_path
