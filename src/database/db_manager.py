"""
Database Manager for AgriFlux Dashboard

Provides SQLite database operations for:
- Processed imagery storage
- Alert management
- AI predictions tracking

Designed for the production-ready dashboard with simple, efficient operations.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for the AgriFlux dashboard.
    """
    
    def __init__(self, db_path: str = "data/agriflux.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_db_directory()
        self._connection = None
        
        # Auto-initialize database schema if it doesn't exist
        try:
            self.init_database()
        except Exception as e:
            logger.warning(f"Could not auto-initialize database: {e}")
        
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            SQLite connection object
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """
        Initialize database schema with all required tables and indexes.
        Creates tables if they don't exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: processed_imagery
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_imagery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    acquisition_date TEXT NOT NULL,
                    tile_id TEXT NOT NULL,
                    cloud_coverage REAL,
                    ndvi_path TEXT,
                    savi_path TEXT,
                    evi_path TEXT,
                    ndwi_path TEXT,
                    ndsi_path TEXT,
                    metadata_json TEXT,
                    processed_at TEXT NOT NULL,
                    synthetic INTEGER DEFAULT 1,
                    UNIQUE(tile_id, acquisition_date)
                )
            """)
            
            # Add synthetic column to existing tables (migration)
            try:
                cursor.execute("""
                    ALTER TABLE processed_imagery 
                    ADD COLUMN synthetic INTEGER DEFAULT 1
                """)
                logger.info("Added synthetic column to processed_imagery table")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            # Table 2: alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    imagery_id INTEGER,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    affected_area TEXT,
                    message TEXT NOT NULL,
                    recommendation TEXT,
                    created_at TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_at TEXT,
                    FOREIGN KEY (imagery_id) REFERENCES processed_imagery(id)
                )
            """)
            
            # Table 3: ai_predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    imagery_id INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predictions_json TEXT NOT NULL,
                    confidence_scores TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (imagery_id) REFERENCES processed_imagery(id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_imagery_date 
                ON processed_imagery(acquisition_date DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_imagery_tile 
                ON processed_imagery(tile_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_imagery_synthetic 
                ON processed_imagery(synthetic)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_imagery 
                ON alerts(imagery_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity 
                ON alerts(severity)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged 
                ON alerts(acknowledged)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_imagery 
                ON ai_predictions(imagery_id)
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    # ========== Processed Imagery Operations ==========
    
    def save_processed_imagery(self, 
                              acquisition_date: str,
                              tile_id: str,
                              cloud_coverage: float,
                              geotiff_paths: Dict[str, str],
                              metadata: Dict[str, Any],
                              synthetic: bool = True) -> int:
        """
        Save processed imagery record to database.
        
        Args:
            acquisition_date: ISO format date string
            tile_id: Sentinel-2 tile identifier
            cloud_coverage: Cloud coverage percentage
            geotiff_paths: Dictionary mapping index names to file paths
            metadata: Additional metadata dictionary
            synthetic: Whether this is synthetic data (default True for backward compatibility)
            
        Returns:
            ID of inserted record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processed_imagery (
                    acquisition_date, tile_id, cloud_coverage,
                    ndvi_path, savi_path, evi_path, ndwi_path, ndsi_path,
                    metadata_json, processed_at, synthetic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                acquisition_date,
                tile_id,
                cloud_coverage,
                geotiff_paths.get('NDVI'),
                geotiff_paths.get('SAVI'),
                geotiff_paths.get('EVI'),
                geotiff_paths.get('NDWI'),
                geotiff_paths.get('NDSI'),
                json.dumps(metadata),
                datetime.now().isoformat(),
                1 if synthetic else 0
            ))
            
            imagery_id = cursor.lastrowid
            data_type = "synthetic" if synthetic else "real"
            logger.info(f"Saved processed imagery record: ID={imagery_id}, tile={tile_id}, type={data_type}")
            return imagery_id
    
    def get_processed_imagery(self, imagery_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve processed imagery record by ID.
        
        Args:
            imagery_id: Imagery record ID
            
        Returns:
            Dictionary with imagery data or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM processed_imagery WHERE id = ?
            """, (imagery_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_latest_imagery(self, tile_id: Optional[str] = None, prefer_real: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the most recent processed imagery record.
        Prioritizes real data over synthetic data by default.
        
        Args:
            tile_id: Optional tile ID filter
            prefer_real: If True, prioritize real data (synthetic=0) over synthetic data
            
        Returns:
            Dictionary with imagery data or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if prefer_real:
                # Try to get real data first
                if tile_id:
                    cursor.execute("""
                        SELECT * FROM processed_imagery 
                        WHERE tile_id = ? AND synthetic = 0
                        ORDER BY acquisition_date DESC 
                        LIMIT 1
                    """, (tile_id,))
                else:
                    cursor.execute("""
                        SELECT * FROM processed_imagery 
                        WHERE synthetic = 0
                        ORDER BY acquisition_date DESC 
                        LIMIT 1
                    """)
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                
                # Fall back to synthetic data if no real data available
                logger.debug("No real data found, falling back to synthetic data")
            
            # Get any data (including synthetic)
            if tile_id:
                cursor.execute("""
                    SELECT * FROM processed_imagery 
                    WHERE tile_id = ?
                    ORDER BY acquisition_date DESC 
                    LIMIT 1
                """, (tile_id,))
            else:
                cursor.execute("""
                    SELECT * FROM processed_imagery 
                    ORDER BY acquisition_date DESC 
                    LIMIT 1
                """)
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def list_processed_imagery(self, 
                              tile_id: Optional[str] = None,
                              limit: int = 50,
                              synthetic: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        List processed imagery records with optional filtering.
        
        Args:
            tile_id: Optional tile ID filter
            limit: Maximum number of records to return
            synthetic: Optional filter - True for synthetic only, False for real only, None for all
            
        Returns:
            List of imagery record dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM processed_imagery WHERE 1=1"
            params = []
            
            if tile_id:
                query += " AND tile_id = ?"
                params.append(tile_id)
            
            if synthetic is not None:
                query += " AND synthetic = ?"
                params.append(1 if synthetic else 0)
            
            query += " ORDER BY acquisition_date DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_temporal_series(self, 
                           tile_id: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           synthetic: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get time series of imagery for temporal analysis.
        
        Args:
            tile_id: Tile identifier
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)
            synthetic: Optional filter - True for synthetic only, False for real only, None for all
            
        Returns:
            List of imagery records ordered by date
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM processed_imagery WHERE tile_id = ?"
            params = [tile_id]
            
            if start_date:
                query += " AND acquisition_date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND acquisition_date <= ?"
                params.append(end_date)
            
            if synthetic is not None:
                query += " AND synthetic = ?"
                params.append(1 if synthetic else 0)
            
            query += " ORDER BY acquisition_date ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_real_imagery(self, 
                        tile_id: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get only real (non-synthetic) imagery records.
        
        Args:
            tile_id: Optional tile ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of real imagery record dictionaries
        """
        return self.list_processed_imagery(tile_id=tile_id, limit=limit, synthetic=False)
    
    def get_synthetic_imagery(self, 
                             tile_id: Optional[str] = None,
                             limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get only synthetic imagery records.
        
        Args:
            tile_id: Optional tile ID filter
            limit: Maximum number of records to return
            
        Returns:
            List of synthetic imagery record dictionaries
        """
        return self.list_processed_imagery(tile_id=tile_id, limit=limit, synthetic=True)
    
    def count_real_imagery(self, tile_id: Optional[str] = None) -> int:
        """
        Count real (non-synthetic) imagery records.
        
        Args:
            tile_id: Optional tile ID filter
            
        Returns:
            Count of real imagery records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if tile_id:
                cursor.execute("""
                    SELECT COUNT(*) FROM processed_imagery 
                    WHERE synthetic = 0 AND tile_id = ?
                """, (tile_id,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM processed_imagery 
                    WHERE synthetic = 0
                """)
            
            return cursor.fetchone()[0]
    
    def count_synthetic_imagery(self, tile_id: Optional[str] = None) -> int:
        """
        Count synthetic imagery records.
        
        Args:
            tile_id: Optional tile ID filter
            
        Returns:
            Count of synthetic imagery records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if tile_id:
                cursor.execute("""
                    SELECT COUNT(*) FROM processed_imagery 
                    WHERE synthetic = 1 AND tile_id = ?
                """, (tile_id,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM processed_imagery 
                    WHERE synthetic = 1
                """)
            
            return cursor.fetchone()[0]
    
    # ========== Alert Operations ==========
    
    def save_alert(self,
                  imagery_id: int,
                  alert_type: str,
                  severity: str,
                  message: str,
                  recommendation: Optional[str] = None,
                  affected_area: Optional[str] = None) -> int:
        """
        Save alert record to database.
        
        Args:
            imagery_id: Associated imagery record ID
            alert_type: Type of alert (e.g., 'vegetation_stress', 'pest_risk')
            severity: Severity level ('critical', 'high', 'medium', 'low')
            message: Alert message
            recommendation: Optional recommendation text
            affected_area: Optional GeoJSON string of affected area
            
        Returns:
            ID of inserted alert
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (
                    imagery_id, alert_type, severity, message,
                    recommendation, affected_area, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                imagery_id,
                alert_type,
                severity,
                message,
                recommendation,
                affected_area,
                datetime.now().isoformat()
            ))
            
            alert_id = cursor.lastrowid
            logger.info(f"Saved alert: ID={alert_id}, type={alert_type}, severity={severity}")
            return alert_id
    
    def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get unacknowledged alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE acknowledged = 0
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_alert_history(self, 
                         imagery_id: Optional[int] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alert history with optional filtering.
        
        Args:
            imagery_id: Optional imagery ID filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if imagery_id:
                cursor.execute("""
                    SELECT * FROM alerts 
                    WHERE imagery_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (imagery_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM alerts 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Mark an alert as acknowledged.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if successful, False if alert not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE alerts 
                SET acknowledged = 1, acknowledged_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), alert_id))
            
            if cursor.rowcount > 0:
                logger.info(f"Acknowledged alert: ID={alert_id}")
                return True
            return False
    
    def get_alert_frequency_by_type(self, days: int = 30) -> Dict[str, int]:
        """
        Get alert frequency by type for the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary mapping alert types to counts
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT alert_type, COUNT(*) as count
                FROM alerts
                WHERE created_at >= ?
                GROUP BY alert_type
                ORDER BY count DESC
            """, (cutoff_date,))
            
            rows = cursor.fetchall()
            return {row['alert_type']: row['count'] for row in rows}
    
    def get_recurring_alerts(self, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Get alerts that have occurred multiple times (recurring patterns).
        
        Args:
            min_occurrences: Minimum number of occurrences to be considered recurring
            
        Returns:
            List of recurring alert patterns
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT alert_type, severity, COUNT(*) as occurrence_count,
                       MIN(created_at) as first_occurrence,
                       MAX(created_at) as last_occurrence
                FROM alerts
                GROUP BY alert_type, severity
                HAVING COUNT(*) >= ?
                ORDER BY occurrence_count DESC
            """, (min_occurrences,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_alerts_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Get alerts filtered by severity level.
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of alert dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE severity = ?
                ORDER BY created_at DESC
            """, (severity,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # ========== AI Predictions Operations ==========
    
    def save_prediction(self,
                       imagery_id: int,
                       model_version: str,
                       prediction_type: str,
                       predictions: Dict[str, Any],
                       confidence_scores: Optional[Dict[str, Any]] = None) -> int:
        """
        Save AI prediction record to database.
        
        Args:
            imagery_id: Associated imagery record ID
            model_version: Model version identifier
            prediction_type: Type of prediction (e.g., 'crop_health', 'yield_forecast')
            predictions: Predictions dictionary
            confidence_scores: Optional confidence scores dictionary
            
        Returns:
            ID of inserted prediction
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ai_predictions (
                    imagery_id, model_version, prediction_type,
                    predictions_json, confidence_scores, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                imagery_id,
                model_version,
                prediction_type,
                json.dumps(predictions),
                json.dumps(confidence_scores) if confidence_scores else None,
                datetime.now().isoformat()
            ))
            
            prediction_id = cursor.lastrowid
            logger.info(f"Saved prediction: ID={prediction_id}, type={prediction_type}")
            return prediction_id
    
    def get_predictions_for_imagery(self, imagery_id: int) -> List[Dict[str, Any]]:
        """
        Get all predictions for a specific imagery record.
        
        Args:
            imagery_id: Imagery record ID
            
        Returns:
            List of prediction dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ai_predictions 
                WHERE imagery_id = ?
                ORDER BY created_at DESC
            """, (imagery_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_latest_prediction(self, 
                            imagery_id: int,
                            prediction_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent prediction for an imagery record and type.
        
        Args:
            imagery_id: Imagery record ID
            prediction_type: Type of prediction
            
        Returns:
            Prediction dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ai_predictions 
                WHERE imagery_id = ? AND prediction_type = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (imagery_id, prediction_type))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    # ========== Utility Operations ==========
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics including real vs synthetic data counts.
        
        Returns:
            Dictionary with database statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            cursor.execute("SELECT COUNT(*) FROM processed_imagery")
            stats['imagery_count'] = cursor.fetchone()[0]
            
            # Count real vs synthetic imagery
            cursor.execute("SELECT COUNT(*) FROM processed_imagery WHERE synthetic = 0")
            stats['real_imagery_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM processed_imagery WHERE synthetic = 1")
            stats['synthetic_imagery_count'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts")
            stats['total_alerts'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
            stats['active_alerts'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ai_predictions")
            stats['predictions_count'] = cursor.fetchone()[0]
            
            # Get date range for all imagery
            cursor.execute("""
                SELECT MIN(acquisition_date), MAX(acquisition_date) 
                FROM processed_imagery
            """)
            date_range = cursor.fetchone()
            stats['date_range'] = {
                'earliest': date_range[0],
                'latest': date_range[1]
            }
            
            # Get date range for real imagery only
            cursor.execute("""
                SELECT MIN(acquisition_date), MAX(acquisition_date) 
                FROM processed_imagery
                WHERE synthetic = 0
            """)
            real_date_range = cursor.fetchone()
            stats['real_date_range'] = {
                'earliest': real_date_range[0],
                'latest': real_date_range[1]
            }
            
            return stats
    
    def clear_all_data(self):
        """
        Clear all data from all tables (use with caution!).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ai_predictions")
            cursor.execute("DELETE FROM alerts")
            cursor.execute("DELETE FROM processed_imagery")
            logger.warning("All data cleared from database")


# Convenience function for quick database initialization
def initialize_database(db_path: str = "data/agriflux.db") -> DatabaseManager:
    """
    Initialize and return a DatabaseManager instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Initialized DatabaseManager instance
    """
    db_manager = DatabaseManager(db_path)
    db_manager.init_database()
    return db_manager
