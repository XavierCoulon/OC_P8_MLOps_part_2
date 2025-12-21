"""Application configuration settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Rugby MLOps"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"
    api_key: str = ""
    hf_repo_id: str = ""
    hf_api_uri: str = ""
    evidently_cloud_token: str = ""
    evidently_project_id: str = ""
    evidently_api_key: str = ""
    database_url: str = ""


settings = Settings()
