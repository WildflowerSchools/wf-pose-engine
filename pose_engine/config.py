from typing import Optional

import platformdirs
import pydantic_settings

APP_NAME = "wf_pose_engine"
APP_AUTHOR = "wildflower"


class Settings(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="POSE_ENGINE_",
    )

    USER_CACHE_DIR: str = platformdirs.user_cache_dir(
        appname=APP_NAME, appauthor=APP_AUTHOR
    )
    RAW_VIDEO_CACHE_DIR: str = f"{USER_CACHE_DIR}/raw_videos"

    RAW_VIDEO_SOURCE_DIR: Optional[str] = None

    MONGO_POSE_URI: str = (
        "mongodb://pose-engine:iamaninsecurepassword@localhost:27017/poses?authSource=poses"
    )

    STORE_POSES: bool = False

    VIDEO_FRAME_LOADER_PROCESSES: Optional[int] = 2

    VIDEO_USE_FFMPEG_WITH_CUDA: bool = (
        False  # Leverage's OPENCV's video_codec;h264_cuvid capture option (only works if proper ffmpeg + GPU encoders are installed)
    )

    VIDEO_USE_CUDACODEC_WITH_CUDA: bool = (
        False  # Leverage's OPENCV's native GPU video processing functionality (only works if proper OpenCV package is installed)
    )
