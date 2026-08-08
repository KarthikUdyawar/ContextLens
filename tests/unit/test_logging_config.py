"""Behavior: configure_logging is idempotent — no duplicate handlers on repeat calls."""
import logging

from src.utils.logging_config import configure_logging


def test_configure_logging_does_not_duplicate_handlers_on_repeat_calls():
    root = logging.getLogger()
    root.handlers.clear()

    configure_logging()
    configure_logging()
    configure_logging()

    assert len(root.handlers) == 1
