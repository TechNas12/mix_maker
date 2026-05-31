import os
import logging
from datetime import datetime
from pathlib import Path
import inspect

class LoggerSetup:
    """
    A utility class to setup and manage logging across the project.
    Creates daily log files with timestamps and function/source information.
    """
    
    def __init__(self, log_folder="logs"):
        """
        Initialize the logger setup.
        
        Args:
            log_folder (str): Path to the folder where logs will be stored. Default is "logs"
        """
        self.log_folder = log_folder
        self._ensure_log_folder()
        self.logger = self._setup_logger()
    
    def _ensure_log_folder(self):
        """Create the logs folder if it doesn't exist."""
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
    
    def _get_log_filename(self):
        """Generate log filename based on current date."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_folder, f"log_{date_str}.txt")
    
    def _setup_logger(self):
        """Setup and configure the logger."""
        logger = logging.getLogger("ProjectLogger")
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        
        # Create a custom formatter with datetime, function name, and message
        formatter = logging.Formatter(
            '%(asctime)s | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with daily rotation
        file_handler = logging.FileHandler(self._get_log_filename(), mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def log(self, message, level="info"):
        """
        Log a message with the caller's function name.
        
        Args:
            message (str): The message to log
            level (str): Log level - 'debug', 'info', 'warning', 'error', 'critical'. Default is 'info'
        """
        # Get the caller's frame information
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        caller_module = caller_frame.f_code.co_filename
        
        # Format the source information
        source_info = f"[{os.path.basename(caller_module)}:{caller_function}]"
        full_message = f"{source_info} {message}"
        
        # Log based on level
        level = level.lower()
        if level == "debug":
            self.logger.debug(full_message)
        elif level == "info":
            self.logger.info(full_message)
        elif level == "warning":
            self.logger.warning(full_message)
        elif level == "error":
            self.logger.error(full_message)
        elif level == "critical":
            self.logger.critical(full_message)
        else:
            self.logger.info(full_message)


# Create a global instance for easy access
_logger_instance = LoggerSetup()


def log(message, level="info"):
    """
    Convenient function to log messages from anywhere in your project.
    
    Usage:
        from logger_utility import log
        
        log("This is an info message")
        log("This is a debug message", level="debug")
        log("This is an error message", level="error")
    
    Args:
        message (str): The message to log
        level (str): Log level - 'debug', 'info', 'warning', 'error', 'critical'. Default is 'info'
    """
    _logger_instance.log(message, level)


def get_logger_instance():
    """
    Get the logger instance if you need direct access.
    
    Returns:
        LoggerSetup: The global logger instance
    """
    return _logger_instance