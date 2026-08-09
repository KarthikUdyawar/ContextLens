"""Centralized logging setup — one handler/format for the whole app."""
import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Set up one StreamHandler + format on the root logger.

    Safe to call more than once — won't stack duplicate handlers.
    Level from `level` arg, else LOG_LEVEL env var, else INFO.
    """
    root = logging.getLogger()
    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO")
    root.setLevel(resolved_level)

    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger — thin pass-through, use after configure_logging()."""
    return logging.getLogger(name)
