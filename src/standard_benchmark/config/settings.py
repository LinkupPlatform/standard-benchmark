"""Centralized configuration using pydantic-settings."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    linkup_api_key: str = Field(default="", alias="LINKUP_API_KEY")
    exa_api_key: str = Field(default="", alias="EXA_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    perplexity_api_key: str = Field(default="", alias="PERPLEXITY_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Configuration
    batch_size: int = Field(default=10, alias="BATCH_SIZE")
    parallel_workers: int = Field(default=4, alias="PARALLEL_WORKERS")

    # Paths
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), alias="CACHE_DIR")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class EvaluationConfig:
    """Load and manage evaluation configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "evaluation_config.yaml"

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        self.weights = data.get("weights", {})
        self.automated_weights = data.get("automated_weights", {})
        self.thresholds = data.get("thresholds", {})


# Global settings instance
settings = Settings()
evaluation_config = EvaluationConfig()
