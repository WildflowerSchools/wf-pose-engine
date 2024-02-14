import platformdirs
import pydantic_settings

APP_NAME = "wf_pose_engine"
APP_AUTHOR = "wildflower"


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    USER_CACHE_DIR: str = platformdirs.user_cache_dir(
        appname=APP_NAME, appauthor=APP_AUTHOR
    )
    RAW_VIDEO_CACHE_DIR: str = f"{USER_CACHE_DIR}/raw_videos"

    MONGO_POSE_URI: str = (
        "mongodb://pose-engine:iamaninsecurepassword@localhost:27017/poses?authSource=poses"
    )
