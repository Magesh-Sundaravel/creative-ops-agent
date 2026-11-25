import logging

from creative_ops.logs import configure_logging, current_brief_id


def test_configure_logging_applies_level_and_puts_brief_id_in_plain_text(capsys):
    root = logging.getLogger()
    old_level, old_handlers = root.level, root.handlers[:]
    try:
        configure_logging("warning")
        token = current_brief_id.set("abc123")
        logging.getLogger("t").warning("hello")
        logging.getLogger("t").info("hidden")
        current_brief_id.reset(token)
        err = capsys.readouterr().err
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)

    assert "[brief=abc123] t: hello" in err
    assert "hidden" not in err
    assert not err.lstrip().startswith("{")
