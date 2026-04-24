"""Runtime configuration. Loaded from environment or .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAMHD_",
        extra="ignore",
    )

    service_name: str = "ramhd"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8100
    log_level: str = "info"


settings = Settings()
