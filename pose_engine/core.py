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


def batch():
    detector_model = DetectorModel.rtmdet_medium()
    pose_model = PoseModel.rtmpose_large_384()

    batch = [
        ["dahlia", "2023-07-20 19:38:33+00:00", "2023-07-20 19:38:35+00:00"],
        ["dahlia", "2023-07-20 19:38:33+00:00", "2023-07-20 19:38:35+00:00"],
        ["dahlia", "2023-07-20 19:38:33+00:00", "2023-07-20 19:38:35+00:00"],
        ["dahlia", "2023-07-20 19:38:33+00:00", "2023-07-20 19:38:35+00:00"],
    ]

    for b in batch:
        environment = b[0]
        start = datetime.fromisoformat(b[1])
        end = datetime.fromisoformat(b[2])

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
