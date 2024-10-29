from pathlib import Path
from typing import Any
import yaml

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_settings import BaseSettings


class EnvironmentConfig(BaseModel):
    NAME: str
    EPISODES: int
    SEED: int | None = None


class Config(BaseSettings):
    ENV: EnvironmentConfig

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def validate_keys(cls, values: dict) -> dict:
        """Convert all dictionary keys to uppercase."""

        def convert_keys(data: dict[str, Any] | list) -> dict:
            if isinstance(data, dict):
                return {key.upper(): convert_keys(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [convert_keys(item) for item in data]

            return data

        return convert_keys(values)


def load_config(filename: Path) -> Config:
    with open(filename, "r") as f:
        yaml_config = yaml.safe_load(f)

    return Config(**yaml_config)
