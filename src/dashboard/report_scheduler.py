"""
Report Scheduler for automated report generation and delivery.
"""

import os
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import time
import threading
from dataclasses import dataclass, asdict
import logging

from .report_generator import ReportGenerator, prepare_report_data


@dataclass
class ScheduledReport:
    """Configuration for a scheduled report"""
    id: str
    name: str
    template: str
    frequency: str  # 'daily', 'weekly', 'monthly'
    time: str  # HH:MM format
    recipients: List[str]
    zones: List[str]  # Zone IDs to include
    enabled: bool = True
    last_generated: Optional[datetime] = None
    next_generation: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.last_generated:
            data['last_generated'] = self.last_generated.isoformat()
        if self.next_generation:
            data['next_generation'] = self.next_generation.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledReport':
        """Create from dictionary"""
        if 'last_generated' in data and data['last_generated']:
            data['last_generated'] = datetime.fromisoformat(data['last_generated'])
        if 'next_generation' in data and data['next_generation']:
            data['next_generation'] = datetime.fromisoformat(data['next_generation'])
        return cls(**data)


class EmailService:
    """Service for sending email notifications and reports"""
    
    def __init__(self, smtp_server: str = "localhost", smtp_port: int = 587,
                 username: Optional[str] = None, password: Optional[str] = None):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
    
    def send_report_email(self, recipients: List[str], subject: str, 
                         body: str, pdf_content: bytes, filename: str) -> bool:
        """
        Send report via email with PDF attachment.
        
        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            body: Email body text
            pdf_content: PDF file content as bytes
            filename: PDF filename
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username or "noreply@agmonitoring.com"
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add PDF attachment
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(pdf_content)
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            msg.attach(attachment)
            
            # Send email
            if self.smtp_server == "localhost":
                # Mock email sending for development
                self.logger.info(f"Mock email sent to {recipients} with subject: {subject}")
                return True
            else:
                # Real email sending
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                text = msg.as_string()
                server.sendmail(msg['From'], recipients, text)
                server.quit()
                
                self.logger.info(f"Email sent successfully to {recipients}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False


class ReportScheduler:
    """Scheduler for automated report generation and delivery"""
    
    def __init__(self, config_file: str = "report_schedules.json"):
        self.config_file = config_file
        self.scheduled_reports: Dict[str, ScheduledReport] = {}
        self.report_generator = ReportGenerator()
        self.email_service = EmailService()
        self.data_provider: Optional[Callable] = None
        self.logger = logging.getLogger(__name__)
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Load existing schedules
        self.load_schedules()
    
    def set_data_provider(self, provider: Callable) -> None:
        """
        Set the data provider function that returns report data.
        
        Args:
            provider: Function that returns (zones, alerts, time_series_data)
        """
        self.data_provider = provider
    
    def add_scheduled_report(self, report: ScheduledReport) -> None:
        """Add a new scheduled report"""
        self.scheduled_reports[report.id] = report
        self._schedule_report(report)
        self.save_schedules()
        self.logger.info(f"Added scheduled report: {report.name}")
    
    def remove_scheduled_report(self, report_id: str) -> bool:
        """Remove a scheduled report"""
        if report_id in self.scheduled_reports:
            report = self.scheduled_reports[report_id]
            # Cancel the scheduled job
            schedule.clear(report_id)
            del self.scheduled_reports[report_id]
            self.save_schedules()
            self.logger.info(f"Removed scheduled report: {report.name}")
            return True
        return False
    
    def update_scheduled_report(self, report: ScheduledReport) -> None:
        """Update an existing scheduled report"""
        if report.id in self.scheduled_reports:
            # Cancel old schedule
            schedule.clear(report.id)
            # Add new schedule
            self.scheduled_reports[report.id] = report
            self._schedule_report(report)
            self.save_schedules()
            self.logger.info(f"Updated scheduled report: {report.name}")
    
    def get_scheduled_reports(self) -> List[ScheduledReport]:
        """Get all scheduled reports"""
        return list(self.scheduled_reports.values())
    
    def get_scheduled_report(self, report_id: str) -> Optional[ScheduledReport]:
        """Get a specific scheduled report"""
        return self.scheduled_reports.get(report_id)
    
    def _schedule_report(self, report: ScheduledReport) -> None:
        """Schedule a report using the schedule library"""
        if not report.enabled:
            return
        
        def job():
            self._generate_and_send_report(report)
        
        # Schedule based on frequency
        if report.frequency == 'daily':
            schedule.every().day.at(report.time).do(job).tag(report.id)
        elif report.frequency == 'weekly':
            schedule.every().monday.at(report.time).do(job).tag(report.id)
        elif report.frequency == 'monthly':
            # Schedule for first day of each month (approximation)
            schedule.every(30).days.at(report.time).do(job).tag(report.id)
        
        # Calculate next generation time
        report.next_generation = self._calculate_next_run(report)
    
    def _calculate_next_run(self, report: ScheduledReport) -> datetime:
        """Calculate the next run time for a report"""
        now = datetime.now()
        time_parts = report.time.split(':')
        hour, minute = int(time_parts[0]), int(time_parts[1])
        
        if report.frequency == 'daily':
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif report.frequency == 'weekly':
            # Next Monday
            days_ahead = 0 - now.weekday()  # Monday is 0
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
        elif report.frequency == 'monthly':
            # First day of next month
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, 
                                     hour=hour, minute=minute, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, 
                                     hour=hour, minute=minute, second=0, microsecond=0)
        else:
            next_run = now + timedelta(days=1)
        
        return next_run
    
    def _generate_and_send_report(self, report: ScheduledReport) -> None:
        """Generate and send a scheduled report"""
        try:
            self.logger.info(f"Generating scheduled report: {report.name}")
            
            # Get data from provider
            if not self.data_provider:
                self.logger.error("No data provider configured")
                return
            
            zones, alerts, time_series_data = self.data_provider()
            
            # Filter data for specified zones if any
            if report.zones:
                zones = [z for z in zones if z.get('id') in report.zones]
                alerts = [a for a in alerts if a.get('zone_id') in report.zones]
                time_series_data = [ts for ts in time_series_data if ts.get('zone_id') in report.zones]
            
            # Prepare report data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Default 30-day period
            
            report_data = prepare_report_data(zones, alerts, time_series_data, start_date, end_date)
            
            # Generate PDF
            pdf_content = self.report_generator.generate_report(report.template, report_data)
            
            # Send email
            subject = f"Agricultural Monitoring Report - {report.name} - {end_date.strftime('%Y-%m-%d')}"
            body = f"""
Dear Recipient,

Please find attached the automated agricultural monitoring report: {report.name}

Report Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This report includes:
- Field condition analysis
- Vegetation index trends
- Active alerts and recommendations
- Zone-specific insights

Best regards,
Agricultural Monitoring System
            """
            
            filename = f"ag_report_{report.name.lower().replace(' ', '_')}_{end_date.strftime('%Y%m%d')}.pdf"
            
            success = self.email_service.send_report_email(
                report.recipients, subject, body, pdf_content, filename
            )
            
            if success:
                report.last_generated = datetime.now()
                report.next_generation = self._calculate_next_run(report)
                self.save_schedules()
                self.logger.info(f"Successfully generated and sent report: {report.name}")
            else:
                self.logger.error(f"Failed to send report: {report.name}")
                
        except Exception as e:
            self.logger.error(f"Error generating scheduled report {report.name}: {e}")
    
    def start_scheduler(self) -> None:
        """Start the report scheduler in a background thread"""
        if self.running:
            return
        
        self.running = True
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Report scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the report scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Report scheduler stopped")
    
    def generate_report_now(self, report_id: str) -> bool:
        """Generate and send a report immediately"""
        report = self.scheduled_reports.get(report_id)
        if not report:
            return False
        
        self._generate_and_send_report(report)
        return True
    
    def save_schedules(self) -> None:
        """Save scheduled reports to file"""
        try:
            data = {
                'reports': [report.to_dict() for report in self.scheduled_reports.values()]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save schedules: {e}")
    
    def load_schedules(self) -> None:
        """Load scheduled reports from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                for report_data in data.get('reports', []):
                    report = ScheduledReport.from_dict(report_data)
                    self.scheduled_reports[report.id] = report
                    self._schedule_report(report)
                
                self.logger.info(f"Loaded {len(self.scheduled_reports)} scheduled reports")
            
        except Exception as e:
            self.logger.error(f"Failed to load schedules: {e}")
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get status information about the scheduler"""
        return {
            'running': self.running,
            'total_reports': len(self.scheduled_reports),
            'enabled_reports': len([r for r in self.scheduled_reports.values() if r.enabled]),
            'next_jobs': [
                {
                    'report_name': report.name,
                    'next_run': report.next_generation.isoformat() if report.next_generation else None
                }
                for report in self.scheduled_reports.values()
                if report.enabled and report.next_generation
            ]
        }


# Utility functions
def create_daily_report_schedule(name: str, recipients: List[str], 
                               zones: List[str], time: str = "08:00") -> ScheduledReport:
    """Create a daily report schedule"""
    return ScheduledReport(
        id=f"daily_{name.lower().replace(' ', '_')}",
        name=name,
        template='standard',
        frequency='daily',
        time=time,
        recipients=recipients,
        zones=zones
    )


def create_weekly_summary_schedule(name: str, recipients: List[str], 
                                 zones: List[str], time: str = "09:00") -> ScheduledReport:
    """Create a weekly summary report schedule"""
    return ScheduledReport(
        id=f"weekly_{name.lower().replace(' ', '_')}",
        name=name,
        template='executive',
        frequency='weekly',
        time=time,
        recipients=recipients,
        zones=zones
    )


def create_monthly_report_schedule(name: str, recipients: List[str], 
                                 zones: List[str], time: str = "10:00") -> ScheduledReport:
    """Create a monthly comprehensive report schedule"""
    return ScheduledReport(
        id=f"monthly_{name.lower().replace(' ', '_')}",
        name=name,
        template='standard',
        frequency='monthly',
        time=time,
        recipients=recipients,
        zones=zones
    )