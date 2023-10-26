import pathlib
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline

from pose_engine.log import logger


class Detector:
    def __init__(self, config: str, checkpoint: str):
        logger.info("Initializing object detector...")

        detector_config = config
        detector_checkpoint = checkpoint
        detector = init_detector(
            config=detector_config, checkpoint=detector_checkpoint, device="cuda:0"
        )
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)

        self.detector = detector
