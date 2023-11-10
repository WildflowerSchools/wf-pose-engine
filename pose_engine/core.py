from datetime import datetime

from .known_inference_models import DetectorModel, PoseModel
from .pipeline import Pipeline


def run(environment: str, start: datetime, end: datetime):
    detector_model = DetectorModel.rtmdet_medium()
    pose_model = PoseModel.rtmpose_large_384()

    p = Pipeline(
        environment=environment,
        start_datetime=start,
        end_datetime=end,
        detector_model=detector_model,
        detector_device="cuda:0",
        pose_estimator_model=pose_model,
        pose_estimator_device="cuda:1",
    )
    p.run()
