from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
import os
import shutil
import tempfile
import time
import uuid

import requests

import video_io
import video_io.client

import pose_engine.config
from pose_engine.log import logger
from pose_engine import util


class VideoFetch:
    def __init__(self):
        self.raw_video_cache_dir = pose_engine.config.Settings().RAW_VIDEO_CACHE_DIR
        self.raw_video_source_dir = pose_engine.config.Settings().RAW_VIDEO_SOURCE_DIR

        self.client: video_io.client.VideoStorageClient = (
            video_io.client.VideoStorageClient(cache_directory=self.raw_video_cache_dir)
        )

    def _load_metadata(
        self,
        start: datetime,
        end: datetime,
        environment_id=None,
        environment_name=None,
    ) -> list[dict]:

        start_datetime = start
        if not isinstance(start, datetime):
            start_datetime = util.str_to_date(start)

        end_datetime = end
        if isinstance(end, datetime):
            end_datetime = util.str_to_date(end)

        def _fetch(retry=0):
            try:
                return video_io.fetch_video_metadata(
                    start=start_datetime,
                    end=end_datetime,
                    environment_id=environment_id,
                    environment_name=environment_name,
                )
            except requests.exceptions.HTTPError as e:
                logger.warning(
                    f"video_io.fetch_video_metadata failed w/ {e.response.status_code} code: {e}"
                )
                if retry >= 3:
                    raise e

                time.sleep(0.5)
                return _fetch(retry + 1)

        videos = _fetch()
        return videos

    def _download_or_copy_from_storage(self, video_metadata: list[dict]):
        video_metadata_with_download_tracking = copy.deepcopy(video_metadata)

        for m in video_metadata_with_download_tracking:
            m["cache_path"] = f"{self.raw_video_cache_dir}/{m['path']}"
            m["cache_file_exists"] = os.path.exists(m["cache_path"])

            raw_source_path = ""
            raw_source_exists = False
            if self.raw_video_source_dir is not None:
                raw_source_path = f"{self.raw_video_source_dir}/{m['path']}"
                raw_source_exists = os.path.exists(m["raw_source_path"])
            m["raw_source_path"] = raw_source_path
            m["raw_source_exists"] = raw_source_exists

        # 1. Try to copy videos from the raw_video_directory if that's available
        def _copy_from_disk(_video_metadata):
            if not os.path.exists(_video_metadata["cache_path"]) and os.path.exists(
                _video_metadata["raw_source_path"]
            ):
                try:
                    shutil.copy(
                        _video_metadata["raw_source_path"],
                        _video_metadata["cache_path"],
                    )
                    logger.info(
                        f"Copied '{_video_metadata['raw_source_path']}' to final storage path '{_video_metadata['cache_path']}' successfully"
                    )
                    _video_metadata["cache_file_exists"] = True
                except Exception:
                    warning = f"Failed copying raw video '{_video_metadata['raw_source_path']}' to final storage path '{_video_metadata['cache_path']}', will attempt to download file using video_io service"
                    logger.warning(warning)

            return _video_metadata

        with ThreadPoolExecutor(max_workers=8) as executor:
            video_metadata_with_download_tracking = list(
                executor.map(_copy_from_disk, video_metadata_with_download_tracking)
            )
            executor.shutdown(wait=True)

        # Pass all video metadata to the video_io.download_video_files, anything that's been cached will not be fetched
        return video_io.download_video_files(
            video_metadata=video_metadata,
            local_video_directory=self.raw_video_cache_dir,
            video_client=self.client,
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

        video_metadata = self._load_metadata(
            environment_id=environment_id,
            environment_name=environment_name,
            start=start,
            end=end,
        )

        video_metadata_post_download = self._download_or_copy_from_storage(
            video_metadata=video_metadata,
        )
        logger.info(f"Finished fetching videos for {environment} from {start} to {end}")

        return video_metadata_post_download
