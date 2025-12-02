"""Application configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "Rugby MLOps API"
    app_version: str = "0.1.0"
    debug: bool = True
    api_prefix: str = "/api/v1"

    class Config:
        """Configuration class."""

        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
