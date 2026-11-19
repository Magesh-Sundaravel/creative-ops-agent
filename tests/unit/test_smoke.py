import pytest
from pydantic import ValidationError

import creative_ops  # noqa: F401
from creative_ops.config import Settings


def test_settings_load(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    settings = Settings()
    assert settings.anthropic_api_key.get_secret_value() == "fake-key"


def test_settings_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
