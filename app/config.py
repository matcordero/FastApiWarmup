from __future__ import annotations
import os
from dataclasses import dataclass

def _split_csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")

@dataclass
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "BassTutor API")
    CORS_ORIGINS: list[str] = None
    MODELS_DIR: str = os.getenv("MODELS_DIR", "/app/models")
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")
    JWT_SECRET_KEY: str | None = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXP_MIN: int = int(os.getenv("JWT_EXP_MIN", str(60 * 24 * 30))) 

    GOOGLE_CLIENT_ID: str | None = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_ID_AUD: str | None = os.getenv("GOOGLE_CLIENT_ID_AUD") 

    MAX_FREE_ANALYSES: int = int(os.getenv("MAX_FREE_ANALYSES", "11"))
    FRONT_ONLY_SECURITY: bool = _env_bool("FRONT_ONLY_SECURITY", True)


    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/app/output")

    def __post_init__(self):
        if not self.DATABASE_URL:
            raise ValueError("ERROR: La variable de entorno DATABASE_URL no está configurada.")
        if not self.JWT_SECRET_KEY:
            raise ValueError("ERROR: La variable de entorno JWT_SECRET_KEY no está configurada.")
        if not self.GOOGLE_CLIENT_ID:
            raise ValueError("ERROR: La variable de entorno GOOGLE_CLIENT_ID no está configurada.")
        
        cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
        self.CORS_ORIGINS = _split_csv(cors_env)


settings = Settings()
APP_NAME = settings.APP_NAME
CORS_ORIGINS = settings.CORS_ORIGINS

DATABASE_URL = settings.DATABASE_URL
JWT_SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ALGORITHM = settings.JWT_ALGORITHM
JWT_EXP_MIN = settings.JWT_EXP_MIN

GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_ID_AUD = settings.GOOGLE_CLIENT_ID_AUD

MAX_FREE_ANALYSES = settings.MAX_FREE_ANALYSES
FRONT_ONLY_SECURITY = settings.FRONT_ONLY_SECURITY

OUTPUT_DIR = settings.OUTPUT_DIR
