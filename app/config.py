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

    outbox_path: str = "data/ramhd_outbox.sqlite"

    state_path: str = "data/linucb_state.json"

    outcome_store_path: str = "data/ramhd_outcomes.sqlite"
 
    # Service auth (Step 8b): shared secret the Node backend sends as the

    # X-RAMHD-Token header. Read from RAMHD_SERVICE_TOKEN (env_prefix above).

    # Empty = auth disabled (local dev); production MUST set it on both services.

    service_token: str = ""
 
 
settings = Settings()

