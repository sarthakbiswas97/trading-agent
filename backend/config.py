"""Configuration management for VAPM."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database (using non-default ports to avoid conflicts)
    database_url: str = "postgresql://postgres:postgres@localhost:5433/vapm"
    redis_url: str = "redis://localhost:6380"

    # Blockchain
    rpc_url: str = "https://sepolia.base.org"
    private_key: str = ""
    etherscan_api_key: str = ""
    chain_id: int = 84532  # Base Sepolia

    # Contract Addresses (deployed on Base Sepolia)
    agent_registry_address: str = ""
    validation_registry_address: str = ""
    trade_executor_address: str = ""

    # Blockchain feature flags
    blockchain_enabled: bool = False  # Set to True when contracts are deployed

    # Market Data
    binance_api_key: str = ""
    binance_secret_key: str = ""

    # Agent Config
    agent_name: str = "VAPM-Alpha"
    max_position_size: float = 0.05  # 5% of capital
    max_daily_loss: float = 0.03     # 3% of capital
    max_drawdown: float = 0.10       # 10% of capital
    trade_interval_seconds: int = 60

    # ML Model
    model_path: str = "ml/models/xgb_v1.joblib"
    prediction_threshold: float = 0.65
    confidence_threshold: float = 0.55

    # API (using 8001 to avoid conflict with other projects)
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        protected_namespaces = ('settings_',)  # Allow model_ prefix


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
