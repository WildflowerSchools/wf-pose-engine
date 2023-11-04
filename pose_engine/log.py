import logging
from logging.config import dictConfig


from pydantic import BaseModel


class LogConfig(BaseModel):
    LOGGER_NAME: str = "wf_pose_engine"
    LOG_FORMAT: str = (
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s"
    )
    LOG_LEVEL: str = "INFO"

    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "format": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers: dict = {
        "wf_pose_engine": {"handlers": ["default"], "level": LOG_LEVEL},
    }


dictConfig(LogConfig().model_dump())
logger = logging.getLogger("wf_pose_engine")
