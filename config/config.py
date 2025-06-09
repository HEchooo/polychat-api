import os

from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    NAME: str = "open-assistant-api"
    DEBUG: bool = False
    ENV: str = "prod"

    BASE_PATH: str = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))

    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8086
    SERVER_WORKERS: int = 8
    API_PREFIX: str = "/api"

    AUTH_ENABLE: bool = False
    AUTH_ADMIN_TOKEN: str = "admin"
    AES_ENCRYPTION_KEY: str = "xxx"

    # Feishu notification settings
    FEISHU_ACCESS_TOKEN: str = ""
    FEISHU_ACCESS_SECRET: str = ""
    FEISHU_ENVIRONMENT: str = "dev"
    FEISHU_ENABLED_ASSISTANTS: str = ""

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
