import logging
import os

def setup_logger(name):
    log_level = logging.DEBUG if os.environ.get("DEBUG_MODE", "False").lower() == "true" else logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler("app.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
