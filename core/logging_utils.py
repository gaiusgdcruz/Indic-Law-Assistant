import logging
import os
import sys
import warnings # Add this import
from logging.handlers import RotatingFileHandler # Optional: if file logging is desired

# Load environment variables for logging configuration
# This assumes .env is loaded by the main application or individual modules' config.
# If not, uncomment the following lines:
# from dotenv import load_dotenv
# dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Assuming .env is in project root
# load_dotenv(dotenv_path=dotenv_path)

DEFAULT_LOG_LEVEL = "INFO"
ENV_LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] [%(processName)s:%(threadName)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Central dictionary to keep track of loggers already configured by this utility
_configured_loggers = {}

# Flag to ensure root logger's basicConfig is only attempted once by this utility
_root_logger_handlers_checked = False

def get_logger(name: str, level: str = None) -> logging.Logger:
    global _root_logger_handlers_checked

    logger_to_configure = logging.getLogger(name)

    # Determine effective log level for this specific logger
    effective_level_str = (level or ENV_LOG_LEVEL).upper()
    numeric_level = getattr(logging, effective_level_str, None)

    if not isinstance(numeric_level, int):
        # Fallback to environment log level, then to default if level string is invalid
        warnings.warn(
            f"Invalid log level string: '{effective_level_str}' for logger '{name}'. "
            f"Defaulting to environment LOG_LEVEL: '{ENV_LOG_LEVEL}'.",
            UserWarning
        )
        numeric_level = getattr(logging, ENV_LOG_LEVEL, logging.INFO) # Default to INFO if ENV_LOG_LEVEL is also bad

    logger_to_configure.setLevel(numeric_level)

    # Configure handlers only if this specific logger instance hasn't been configured by this utility before.
    # This prevents adding duplicate handlers to the same logger instance if get_logger is called multiple times for it.
    if name not in _configured_loggers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        
        # Remove any existing handlers from this logger before adding new ones
        # to prevent duplication if it was somehow configured by other means.
        # Be careful with this if other parts of the system might legitimately add handlers.
        # For a central utility, taking full control is often desired.
        if logger_to_configure.hasHandlers():
            logger_to_configure.handlers.clear()
            
        logger_to_configure.addHandler(console_handler)

        # Optional: Add a rotating file handler for all loggers obtained through this utility
        # Ensure LOG_FILE_PATH is defined in .env or has a sensible default
        log_file_path = os.getenv("APP_LOG_FILE_PATH", os.path.join(os.path.dirname(__file__), "..", "app.log"))
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Rotating File Handler
        # Max 10MB per file, keep 5 backup files
        file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger_to_configure.addHandler(file_handler)

        # Mark this logger as configured by this utility
        _configured_loggers[name] = logger_to_configure
        
        # By default, loggers propagate messages to handlers of ancestor loggers.
        # Set to False if you want this logger to only use its own handlers and not pass messages up.
        # logger_to_configure.propagate = False


    # Ensure that the root logger has at least one handler if no handlers are configured for it yet.
    # This helps catch logs from third-party libraries that log to the root logger.
    # This check and configuration should ideally run only once.
    if not _root_logger_handlers_checked and not logging.getLogger().hasHandlers():
        # Temporarily get the root logger to configure it.
        # Using get_logger('') or logging.getLogger()
        root_logger_instance = logging.getLogger()
        root_logger_instance.setLevel(ENV_LOG_LEVEL) # Set root logger level from env or default

        # Add console handler to root if none exist
        root_console_handler = logging.StreamHandler(sys.stdout)
        root_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        root_console_handler.setFormatter(root_formatter)
        root_logger_instance.addHandler(root_console_handler)
        _root_logger_handlers_checked = True


    return logger_to_configure
