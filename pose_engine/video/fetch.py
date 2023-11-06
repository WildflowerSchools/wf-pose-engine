from datetime import datetime
import uuid

import video_io
import video_io.client

import pose_engine.config
from pose_engine.log import logger


class VideoFetch:
    def __init__(self):
        self.client: video_io.client.VideoStorageClient = (
            video_io.client.VideoStorageClient(
                cache_directory=pose_engine.config.Settings().RAW_VIDEO_CACHE_DIR
            )
        )

    def fetch(
        self,
        environment: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        logger.info(f"Fetching videos for {environment} from {start} to {end}...")
        environment_id = None
        environment_name = None
        try:
            uuid.UUID(str(environment))
            environment_id = environment
        except ValueError:
            environment_name = environment

        videos = video_io.fetch_videos(
            video_client=self.client,
            local_video_directory=self.client.CACHE_DIRECTORY,
            environment_id=environment_id,
            environment_name=environment_name,
            start=start,
            end=end,
        )
        logger.info(f"Finished fetching videos for {environment} from {start} to {end}")

        return videos
