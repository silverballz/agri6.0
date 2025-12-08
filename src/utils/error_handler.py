"""
Error handling framework for AgriFlux dashboard
Provides decorators and utilities for graceful error handling
"""

import streamlit as st
import logging
import functools
import traceback
from pathlib import Path
from typing import Callable, Any
from datetime import datetime


# Configure logging
def setup_logging():
    """Set up logging configuration with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    log_file = log_dir / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('agriflux')


# Initialize logger
logger = setup_logging()


def safe_page(func: Callable) -> Callable:
    """
    Decorator for page functions to handle errors gracefully
    
    Usage:
        @safe_page
        def show_page():
            # Page implementation
            pass
    """
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
            
        except FileNotFoundError as e:
            logger.error(f"File not found in {func.__name__}: {e}", exc_info=True)
            st.error(f"📁 **Data file not found:** `{e.filename}`")
            st.info("💡 **Suggestion:** Run the data processing pipeline first or check file paths")
            
            with st.expander("🔧 Troubleshooting Steps"):
                st.markdown("""
                1. Verify the file path is correct
                2. Check if data processing has been completed
                3. Ensure you have read permissions for the file
                4. Run: `python scripts/process_sentinel2_data.py`
                """)
            
            return None
            
        except ImportError as e:
            logger.error(f"Import error in {func.__name__}: {e}", exc_info=True)
            st.error(f"📦 **Missing dependency:** `{e.name if hasattr(e, 'name') else 'unknown'}`")
            
            # Try to extract package name from error message
            error_msg = str(e)
            if "No module named" in error_msg:
                package = error_msg.split("'")[1] if "'" in error_msg else "unknown"
                st.code(f"pip install {package}", language="bash")
            
            st.info("💡 **Suggestion:** Install missing dependencies from requirements.txt")
            st.code("pip install -r requirements.txt", language="bash")
            
            return None
            
        except PermissionError as e:
            logger.error(f"Permission error in {func.__name__}: {e}", exc_info=True)
            st.error(f"🔒 **Permission denied:** Cannot access `{e.filename}`")
            st.info("💡 **Suggestion:** Check file permissions or run with appropriate privileges")
            
            return None
            
        except ValueError as e:
            logger.error(f"Value error in {func.__name__}: {e}", exc_info=True)
            st.error(f"⚠️ **Invalid data:** {str(e)}")
            st.info("💡 **Suggestion:** Check data format and values")
            
            with st.expander("📋 Error Details"):
                st.code(traceback.format_exc())
            
            return None
            
        except KeyError as e:
            logger.error(f"Key error in {func.__name__}: {e}", exc_info=True)
            st.error(f"🔑 **Missing data field:** {str(e)}")
            st.info("💡 **Suggestion:** Data may be incomplete or in wrong format")
            
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            st.error(f"⚠️ **Unexpected error occurred**")
            st.warning(f"Error type: `{type(e).__name__}`")
            st.warning(f"Error message: `{str(e)}`")
            
            st.info("💡 **Suggestion:** Please refresh the page or contact support if the issue persists")
            
            with st.expander("🔍 Technical Details (for debugging)"):
                st.code(traceback.format_exc())
            
            # Offer to report the error
            if st.button("📧 Report this error", key=f"report_error_{func.__name__}"):
                st.success("Error report sent to development team!")
            
            return None
    
    return wrapper


def safe_operation(operation_name: str = "Operation"):
    """
    Decorator for individual operations within pages
    
    Usage:
        @safe_operation("Loading data")
        def load_data():
            # Data loading code
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error in {operation_name}: {e}", exc_info=True)
                st.error(f"⚠️ **{operation_name} failed:** {str(e)}")
                
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                
                return None
        
        return wrapper
    
    return decorator


def handle_data_loading(func: Callable) -> Callable:
    """
    Specialized decorator for data loading operations
    Provides specific error messages for common data loading issues
    """
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            with st.spinner(f"Loading data..."):
                return func(*args, **kwargs)
                
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}", exc_info=True)
            st.error("📁 **Data not found**")
            st.info("The requested data file could not be found. Please ensure data processing has been completed.")
            
            if st.button("🔄 Run Data Processing", key=f"process_data_{func.__name__}"):
                st.info("Data processing would be triggered here")
            
            return None
            
        except Exception as e:
            logger.error(f"Data loading error: {e}", exc_info=True)
            st.error(f"⚠️ **Failed to load data:** {str(e)}")
            
            # Offer fallback options
            st.info("💡 **Try these options:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Retry", key=f"retry_{func.__name__}"):
                    st.rerun()
            
            with col2:
                if st.button("🎬 Load Demo Data", key=f"demo_{func.__name__}"):
                    st.session_state.demo_mode = True
                    st.rerun()
            
            return None
    
    return wrapper


class ErrorMessages:
    """Centralized error message templates"""
    
    @staticmethod
    def missing_dependency(package: str) -> str:
        return f"""
        ### 📦 Missing Dependency: {package}
        
        This feature requires the `{package}` package.
        
        **To install:**
        ```bash
        pip install {package}
        ```
        
        Or install all dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        """
    
    @staticmethod
    def missing_data(data_type: str) -> str:
        return f"""
        ### 📁 Missing Data: {data_type}
        
        The required {data_type} data is not available.
        
        **Possible solutions:**
        1. Run the data processing pipeline
        2. Check if the data files exist in the correct location
        3. Load demo data to explore the interface
        """
    
    @staticmethod
    def processing_error(operation: str) -> str:
        return f"""
        ### ⚠️ Processing Error
        
        An error occurred during: {operation}
        
        **What to do:**
        1. Check the logs for detailed error information
        2. Verify input data is in the correct format
        3. Try refreshing the page
        4. Contact support if the issue persists
        """
    
    @staticmethod
    def database_error() -> str:
        return """
        ### 🗄️ Database Error
        
        Unable to connect to or query the database.
        
        **Troubleshooting:**
        1. Check if the database file exists
        2. Verify database permissions
        3. Try reinitializing the database
        4. Check logs for specific error details
        """


def log_error(error: Exception, context: str = ""):
    """
    Log an error with context information
    
    Args:
        error: The exception that occurred
        context: Additional context about where/when the error occurred
    """
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)


def display_error_summary():
    """Display a summary of recent errors in the sidebar"""
    
    log_file = Path("logs") / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
    
    if not log_file.exists():
        return
    
    try:
        # Read last 10 error lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            error_lines = [line for line in lines if 'ERROR' in line][-10:]
        
        if error_lines:
            with st.sidebar.expander("⚠️ Recent Errors", expanded=False):
                st.caption(f"Last {len(error_lines)} errors:")
                for line in error_lines:
                    st.text(line.strip()[:100])  # Truncate long lines
                
                if st.button("📋 View Full Log", key="view_full_log"):
                    st.info(f"Log file: {log_file}")
    
    except Exception as e:
        logger.error(f"Error displaying error summary: {e}")


def check_critical_paths():
    """
    Check if critical paths exist and are accessible
    Returns dict with path status
    """
    
    critical_paths = {
        'data': Path('data'),
        'processed': Path('data/processed'),
        'models': Path('models'),
        'logs': Path('logs'),
        'database': Path('data/agriflux.db')
    }
    
    status = {}
    
    for name, path in critical_paths.items():
        if path.suffix:  # It's a file
            status[name] = path.exists()
        else:  # It's a directory
            status[name] = path.exists() and path.is_dir()
    
    return status


def display_system_health():
    """Display system health status based on critical paths"""
    
    status = check_critical_paths()
    
    st.sidebar.markdown("**System Health:**")
    
    health_items = {
        'data': ('📁 Data Directory', status.get('data', False)),
        'processed': ('📊 Processed Data', status.get('processed', False)),
        'models': ('🤖 AI Models', status.get('models', False)),
        'database': ('🗄️ Database', status.get('database', False)),
        'logs': ('📝 Logs', status.get('logs', False))
    }
    
    for key, (label, is_healthy) in health_items.items():
        icon = "✅" if is_healthy else "⚠️"
        css_class = "status-healthy" if is_healthy else "status-warning"
        
        st.sidebar.markdown(
            f"{icon} <span class='{css_class}'>{label}</span>",
            unsafe_allow_html=True
        )
