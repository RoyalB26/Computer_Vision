"""Logging utilities for training and evaluation."""
import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "BaseModel",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Initialize and configure a logger with formatted console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Filepath to save the log output.
        log_level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Avoid duplicate handlers if logger was already created
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is not None:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
