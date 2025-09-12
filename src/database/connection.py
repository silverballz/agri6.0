"""
Database connection management for SQLite with spatial extensions.
"""

import sqlite3
import os
from typing import Optional, Any, Dict, List
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages SQLite database connections with spatial extensions.
    """
    
    def __init__(self, db_path: str = "agricultural_monitoring.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for database operations."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def connect(self) -> sqlite3.Connection:
        """
        Create and return a database connection with spatial extensions.
        
        Returns:
            SQLite connection object
        """
        if self._connection is None:
            try:
                self._connection = sqlite3.connect(
                    self.db_path,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
                )
                
                # Enable foreign key constraints
                self._connection.execute("PRAGMA foreign_keys = ON")
                
                # Try to load SpatiaLite extension (optional for development)
                try:
                    self._connection.enable_load_extension(True)
                    # Common SpatiaLite library locations
                    spatialite_paths = [
                        'mod_spatialite',
                        'mod_spatialite.so',
                        'libspatialite.so',
                        '/usr/lib/x86_64-linux-gnu/mod_spatialite.so',
                        '/usr/local/lib/mod_spatialite.so'
                    ]
                    
                    spatialite_loaded = False
                    for path in spatialite_paths:
                        try:
                            self._connection.load_extension(path)
                            spatialite_loaded = True
                            logger.info(f"SpatiaLite extension loaded from {path}")
                            break
                        except sqlite3.OperationalError:
                            continue
                    
                    if not spatialite_loaded:
                        logger.warning("SpatiaLite extension not found. Spatial operations will be limited.")
                    
                    self._connection.enable_load_extension(False)
                    
                except sqlite3.OperationalError as e:
                    logger.warning(f"Could not load SpatiaLite extension: {e}")
                
                # Set row factory for easier data access
                self._connection.row_factory = sqlite3.Row
                
                logger.info(f"Connected to database: {self.db_path}")
                
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database: {e}")
                raise
        
        return self._connection
    
    def disconnect(self):
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    @contextmanager
    def get_cursor(self):
        """
        Context manager for database cursor operations.
        
        Yields:
            SQLite cursor object
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        
        Yields:
            SQLite connection object
        """
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name=?
        """
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get the schema information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        query = f"PRAGMA table_info({table_name})"
        with self.get_cursor() as cursor:
            cursor.execute(query)
            columns = cursor.fetchall()
            
        return [
            {
                'name': col['name'],
                'type': col['type'],
                'not_null': bool(col['notnull']),
                'default_value': col['dflt_value'],
                'primary_key': bool(col['pk'])
            }
            for col in columns
        ]
    
    def backup_database(self, backup_path: str):
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file
        """
        conn = self.connect()
        try:
            with sqlite3.connect(backup_path) as backup_conn:
                conn.backup(backup_conn)
            logger.info(f"Database backed up to: {backup_path}")
        except sqlite3.Error as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get general information about the database.
        
        Returns:
            Dictionary with database information
        """
        info = {
            'db_path': self.db_path,
            'file_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0,
            'tables': []
        }
        
        # Get list of tables
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = self.execute_query(query)
        
        for table in tables:
            table_name = table['name']
            # Get row count for each table
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = self.execute_query(count_query)
            row_count = count_result[0]['count'] if count_result else 0
            
            info['tables'].append({
                'name': table_name,
                'row_count': row_count
            })
        
        return info
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()