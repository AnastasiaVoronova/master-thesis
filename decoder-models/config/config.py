from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_api_key: str
    model_api_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
