"""Logging configuration and utilities."""

import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Configure logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # File handler
    file_handler = RotatingFileHandler(
        "app.log", maxBytes=10485760, backupCount=5  # 10MB
    )
    file_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger(__name__)
