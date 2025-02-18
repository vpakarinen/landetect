import logging.config
import logging
import os

def setup_logger(name=None):
    """Configure logging for the entire application using a dictionary configuration."""
    logger = logging.getLogger(name) if name else logging.getLogger()
    if logger.hasHandlers():
        return logger

    log_level = logging.DEBUG if os.environ.get("DEBUG_MODE", "False").lower() == "true" else logging.INFO

    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)

    config = {
        'version': 1,
        'disable_existing_loggers': False,
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
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.info("Logger configured, logging to file 'logs/app.log'")
    return logger
