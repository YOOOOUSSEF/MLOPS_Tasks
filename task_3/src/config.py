from pathlib import Path
import yaml
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: Path) -> dict:
    logger.info("Loading config from: %s", config_path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        logger.info("Config loaded successfully")
        return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        raise
    except Exception as exc:
        logger.exception("Failed to load config: %s", exc)
        raise


def get_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    logger.info("Project root resolved to: %s", project_root)

    return project_root
