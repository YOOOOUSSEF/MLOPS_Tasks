from pathlib import Path
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]