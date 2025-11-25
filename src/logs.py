import logging
import os
from datetime import datetime
from typing import Optional


LOG_FILENAME = "log.txt"


def setup_logger(output_dir: str, name: str = "pipeline") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers during repeated calls
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(LOG_FILENAME) for h in logger.handlers):
        handler = logging.FileHandler(os.path.join(output_dir, LOG_FILENAME))
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_parameters(logger: logging.Logger, pixel_mm_ratio: float, tail_factor: float) -> None:
    logger.info("PARAMETERS | pixel_mm_ratio=%s tail_factor=%s", pixel_mm_ratio, tail_factor)


def log_status(logger: Optional[logging.Logger], message: str) -> None:
    if logger:
        logger.info(message)


def log_error(logger: Optional[logging.Logger], message: str) -> None:
    if logger:
        logger.error(message)


def timestamp_now() -> str:
    return datetime.utcnow().isoformat()

