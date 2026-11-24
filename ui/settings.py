from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    """The UI needs only where the API is; it never holds the Anthropic key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_url: str = "http://localhost:8000"
