"""Behavior: configure_logging is idempotent — no duplicate handlers on repeat calls."""
import logging

from src.utils.logging_config import configure_logging


def test_configure_logging_does_not_duplicate_handlers_on_repeat_calls():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        root.handlers.clear()
        configure_logging()
        configure_logging()
        configure_logging()

        assert len(root.handlers) == 1
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
