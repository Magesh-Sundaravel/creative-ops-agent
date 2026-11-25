"""Plain-text logging that carries the current Brief id on every line."""

import logging
from contextvars import ContextVar

current_brief_id: ContextVar[str] = ContextVar("current_brief_id", default="-")

_factory_installed = False


def configure_logging(level: str) -> None:
    """Apply the log level and a plain-text format showing the Brief id."""
    global _factory_installed
    if not _factory_installed:
        factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs) -> logging.LogRecord:
            record = factory(*args, **kwargs)
            record.brief_id = current_brief_id.get()
            return record

        logging.setLogRecordFactory(record_factory)
        _factory_installed = True
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [brief=%(brief_id)s] %(name)s: %(message)s",
        force=True,
    )
