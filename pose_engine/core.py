from datetime import datetime, timedelta

import pandas as pd
import torch.multiprocessing as mp

from .handle.handle import Pose2dHandle
from .honeycomb_service import HoneycombCachingClient
from .known_inference_models import DetectorModel, PoseModel
from .log import logger
from .pipeline import Pipeline


def run(environment: str, start: datetime, end: datetime):
    detector_model = DetectorModel.rtmdet_medium()
    pose_model = PoseModel.rtmpose_large_384()

    process_manager = mp.Manager()

    p = Pipeline(
        environment=environment,
        start_datetime=start,
        end_datetime=end,
        mp_manager=process_manager,
        detector_model=detector_model,
        detector_device="cuda:0",
        pose_estimator_model=pose_model,
        pose_estimator_device="cuda:1",
    )
    p.run()


def batch():
    detector_model = DetectorModel.rtmdet_medium()
    pose_model = PoseModel.rtmpose_large_384()

    honeycomb_client = HoneycombCachingClient()

    poses = Pose2dHandle().fetch_current_pose_coverage()

    batch = [
        ["dahlia", "2023-06-30 20:32:46+00:00", "2023-06-30 20:34:58+00:00"],
        ["dahlia", "2023-06-30 20:35:14+00:00", "2023-06-30 20:43:45+00:00"],
        ["dahlia", "2023-06-30 20:44:59+00:00", "2023-06-30 21:34:00+00:00"],
        ["dahlia", "2023-06-30 21:37:54+00:00", "2023-06-30 21:47:05+00:00"],
        ["dahlia", "2023-07-20 19:31:48+00:00", "2023-07-20 19:33:14+00:00"],
        ["dahlia", "2023-07-20 19:33:33+00:00", "2023-07-20 19:42:52+00:00"],
        ["dahlia", "2023-07-20 19:43:25+00:00", "2023-07-20 19:47:25+00:00"],
        ["dahlia", "2023-07-20 19:58:09+00:00", "2023-07-20 20:02:46+00:00"],
        ["dahlia", "2023-07-20 20:25:10+00:00", "2023-07-20 20:28:49+00:00"],
        ["dahlia", "2023-07-20 20:46:29+00:00", "2023-07-20 21:03:02+00:00"],
        ["dahlia", "2023-07-20 21:05:21+00:00", "2023-07-20 21:09:50+00:00"],
        ["dahlia", "2023-07-20 21:10:33+00:00", "2023-07-20 21:35:21+00:00"],
        ["dahlia", "2023-07-20 21:43:39+00:00", "2023-07-20 21:53:20+00:00"],
        ["dahlia", "2023-07-20 21:59:33+00:00", "2023-07-20 22:07:21+00:00"],
        ["dahlia", "2023-07-20 22:27:51+00:00", "2023-07-20 22:39:33+00:00"],
        ["dahlia", "2023-07-20 22:42:25+00:00", "2023-07-20 22:43:58+00:00"],
        ["dahlia", "2023-07-20 22:57:15+00:00", "2023-07-20 23:07:14+00:00"],
        ["dahlia", "2023-07-20 23:12:27+00:00", "2023-07-20 23:17:14+00:00"],
        ["dahlia", "2023-07-20 23:17:15+00:00", "2023-07-20 23:17:40+00:00"],
        ["dahlia", "2023-07-20 23:18:03+00:00", "2023-07-20 23:25:51+00:00"],
    ]
    batch = list(
        map(
            lambda v: [
                v[0],
                datetime.fromisoformat(v[1]),
                datetime.fromisoformat(v[2]),
            ],
            batch,
        )
    )

    existing_inferences = list(
        map(lambda p: [p["environment_id"], p["start"], p["end"]], poses)
    )
    for e in existing_inferences:
        e[0] = honeycomb_client.fetch_environment(environment=e[0])["environment_name"]

    df_existing_inferences = pd.DataFrame(
        existing_inferences, columns=["environment", "start_datetime", "end_datetime"]
    )

    df_batch = pd.DataFrame(
        batch,
        columns=["environment", "start_datetime", "end_datetime"],
    )
    df_batch["start_datetime"] = df_batch["start_datetime"] - timedelta(seconds=10)
    df_batch["end_datetime"] = df_batch["end_datetime"] + timedelta(seconds=10)

    def merge_batch(_df_batch):
        segments = []
        for _, segment in _df_batch.iterrows():
            environment = segment["environment"]
            start = segment["start_datetime"]
            end = segment["end_datetime"]
            if len(segments) == 0:
                segments.append(
                    {
                        "environment": environment,
                        "start_datetime": start,
                        "end_datetime": end,
                    }
                )
                continue

            merged = False
            for s in segments:
                if environment != s["environment"]:
                    continue

                if start < s["start_datetime"] and (
                    end >= s["start_datetime"] and end <= s["end_datetime"]
                ):
                    s["start_datetime"] = start
                    merged = True
                    break
                elif end > s["end_datetime"] and (
                    start >= s["start_datetime"] and start <= s["end_datetime"]
                ):
                    s["end_datetime"] = end
                    merged = True
                    break
                elif start >= s["start_datetime"] and end <= s["end_datetime"]:
                    merged = True
                    break

            if not merged:
                segments.append(
                    {
                        "environment": environment,
                        "start_datetime": start,
                        "end_datetime": end,
                    }
                )

        df_merged_segments = pd.DataFrame(segments).sort_values(by=["start_datetime"])
        return df_merged_segments

    df_existing_merged_segments = merge_batch(df_existing_inferences)
    df_batch_merged_segments = merge_batch(df_batch)

    segments = []
    for (
        environment_name,
        df_batch_merged_segments_by_environment,
    ) in df_batch_merged_segments.groupby(by="environment"):
        df_existing_merged_segments_by_environment = df_existing_merged_segments[
            df_existing_merged_segments["environment"] == environment_name
        ]

        existing_time_set = set()
        for _, row in df_existing_merged_segments_by_environment.iterrows():
            set_date_range = set(
                pd.date_range(
                    start=row["start_datetime"], end=row["end_datetime"], freq="100L"
                )
            )
            existing_time_set = existing_time_set.union(set_date_range)

        new_time_set = set()
        for _, row in df_batch_merged_segments_by_environment.iterrows():
            set_date_range = set(
                pd.date_range(
                    start=row["start_datetime"], end=row["end_datetime"], freq="100L"
                )
            )
            new_time_set = new_time_set.union(set_date_range)

        valid_times = new_time_set - existing_time_set
        df_valid_times = (
            pd.DataFrame(valid_times, columns=["frame_time"])
            .sort_values(by="frame_time")
            .reset_index(drop=True)
        )
        df_valid_times["gap"] = df_valid_times["frame_time"].diff() > pd.to_timedelta(
            "100L"
        )

        df_valid_times["group"] = df_valid_times["gap"].cumsum()
        for _, df_valid_group in df_valid_times.groupby(by="group"):
            segments.append(
                {
                    "environment": environment_name,
                    "start_datetime": df_valid_group["frame_time"].min(),
                    "end_datetime": df_valid_group["frame_time"].max(),
                }
            )
    df_merged_segments = pd.DataFrame(segments)

    process_manager = mp.Manager()

    p = Pipeline(
        mp_manager=process_manager,
        detector_model=detector_model,
        detector_device="cuda:0",
        pose_estimator_model=pose_model,
        pose_estimator_device="cuda:1",
        use_fp_16=False,
    )

    df_merged_segments["length"] = (
        df_merged_segments["end_datetime"] - df_merged_segments["start_datetime"]
    )

    logger.info(
        f"Starting pose processing, working through {df_merged_segments['length'].sum()} of the classroom"
    )
    for ii, (_, b) in enumerate(df_merged_segments.iterrows()):
        logger.info(f"Processing classroom segment {ii+1}/{len(df_merged_segments)}")
        environment = b["environment"]
        start = b["start_datetime"].to_pydatetime()
        end = b["end_datetime"].to_pydatetime()

        p.run(environment=environment, start_datetime=start, end_datetime=end)
