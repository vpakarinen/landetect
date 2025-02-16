import logging
import os
import logging.config

def setup_logger(name=None):
    """
    Configure logging for the entire application using a dictionary configuration with dictConfig.
    This ensures that all log messages, including those from child loggers, are handled by the
    configured console and file handlers. Existing loggers are not disabled, so they will propagate
    their messages to the root logger.
    """
    log_level = logging.DEBUG if os.environ.get("DEBUG_MODE", "False").lower() == "true" else logging.INFO

    # Ensure the "logs" directory exists
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)

    config = {
        'version': 1,
        'disable_existing_loggers': False,  # Do not disable existing loggers
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'filename': 'logs/app.log',
                'mode': 'a',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': log_level,
        },
    }

    logging.config.dictConfig(config)
    
    # Get the requested logger; if a child logger is created later, it will propagate to the root logger.
    logger = logging.getLogger() if name is None else logging.getLogger(name)
    logger.info("Logger configured, logging to file 'logs/app.log'")
    return logger
