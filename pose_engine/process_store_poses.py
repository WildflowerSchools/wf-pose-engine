from datetime import datetime
from typing import Optional

import torch.multiprocessing as mp

from . import loaders
from .handle import Pose2dHandle
from .handle.models.pose_2d import (
    Pose2d,
    Pose2dMetadata,
    Pose2dMetadataCommon,
    PoseOutput,
    BoundingBoxOutput,
)
from .log import logger


class ProcessStorePoses:
    def __init__(self, input_poses_loader: loaders.PosesDataLoader):
        self.input_poses_loader: loaders.PosesDataLoader = input_poses_loader
        self.process: Optional[mp.Process] = None

    def start(self, common_metadata: Pose2dMetadataCommon):
        if self.process is None:
            self.process = mp.Process(target=self._run, args=(common_metadata,))
            self.process.start()

    def wait(self):
        self.process.join()
        logger.info(f"ProcessStorePoses service finished: {self.process.exitcode}")

    def stop(self):
        self.process.close()
        self.process = None

    def mark_poses_drained(self):
        self.input_poses_loader.dataset.done_loading()

    def _run(self, common_metadata: Pose2dMetadataCommon):
        logger.info("Running ProcessStorePoses service...")
        mongo_handle = Pose2dHandle()

        for _batch_idx, (poses, bboxes, meta) in enumerate(self.input_poses_loader):
            pose_2d_batch = []
            for idx, pose in enumerate(poses):
                pose_box = bboxes[idx]
                pose_meta = meta[idx]

                timestamp = datetime.utcfromtimestamp(
                    float(pose_meta["frame_timestamp"])
                )
                metadata = Pose2dMetadata(
                    **common_metadata.model_dump(),
                    camera_device_id=pose_meta["camera_device_id"],
                )

                # if pose_meta["camera_device_id"] == "c9f013f9-3100-4c2f-9762-c1fb35b445a0":
                #     logger.info(f"Found pose at {timestamp}")

                pose_2d = Pose2d(
                    timestamp=timestamp,
                    pose=PoseOutput(keypoints=pose),
                    bbox=BoundingBoxOutput(bbox=pose_box),
                    metadata=metadata,
                )
                pose_2d_batch.append(pose_2d)

            mongo_handle.insert_poses(pose_2d_batch)

        logger.info("ProcessStorePoses service loop ended")
