from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import torch.multiprocessing as mp

from pose_db_io import PoseHandle

from . import config
from .honeycomb_service import HoneycombCachingClient
from .known_inference_models import DetectorModel, PoseModel
from .log import logger
from .pipeline import Pipeline


def _run(
    environment: str,
    start: datetime,
    end: datetime,
    detector_model: Optional[DetectorModel],
    pose_model: Optional[PoseModel],
    run_parallel: bool = False,
    run_distributed: bool = False,
    use_fp16: bool = True,
    compile_engine: Optional[str] = None,
    detector_batch_size: Optional[int] = None,
    pose_estimator_batch_size: Optional[int] = None,
):
    p = Pipeline(
        detector_model=detector_model,
        detector_device="cuda:0",  # Ignored when using run_distributed
        pose_estimator_model=pose_model,
        pose_estimator_device="cuda:0",  # Ignored when using run_distributed
        use_fp16=use_fp16,
        run_parallel=run_parallel,
        run_distributed=run_distributed,
        compile_engine=compile_engine,
        detector_batch_size=detector_batch_size,
        pose_estimator_batch_size=pose_estimator_batch_size,
    )
    p.run(
        environment=environment,
        start_datetime=start,
        end_datetime=end,
    )


def run(
    environment: str,
    start: datetime,
    end: datetime,
    detector_batch_size: Optional[int] = None,
    pose_estimator_batch_size: Optional[int] = None,
    use_fp16: bool = True,
    compile_engine: Optional[str] = None,
):
    run_parallel = False

    ###########################
    # Standard top-down model
    ###########################
    # run_distributed = False
    # detector_batch_size = 110
    # pose_estimator_batch_size = 350
    # detector_model = DetectorModel.rtmdet_medium()
    # pose_model = PoseModel.rtmpose_large_384()

    ###########################
    # TensorRT top-down model
    ###########################
    # run_distributed = False
    # detector_batch_size = 110
    # pose_estimator_batch_size = 350
    # detector_model = DetectorModel.rtmdet_medium_tensorrt_dynamic_640x640_batch()
    # detector_model = DetectorModel.rtmdet_medium_tensorrt_dynamic_640x640_fp16_batch()
    # pose_model = PoseModel.rtmpose_large_384_tensorrt_batch()

    ###########################
    # One-stage model (RTMPose Large)
    ###########################
    run_distributed = True
    detector_model = None
    # detector_batch_size = None
    # pose_estimator_batch_size = 118
    pose_model = PoseModel.rtmo_large()

    ###########################
    # One-stage model (RTMPose Medium)
    ###########################
    # run_distributed = True
    # detector_model = None
    # detector_batch_size = None
    # pose_estimator_batch_size = 146
    # pose_model = PoseModel.rtmo_medium()

    _run(
        environment=environment,
        start=start,
        end=end,
        detector_model=detector_model,
        pose_model=pose_model,
        run_parallel=run_parallel,
        run_distributed=run_distributed,
        detector_batch_size=detector_batch_size,
        pose_estimator_batch_size=pose_estimator_batch_size,
        compile_engine=compile_engine,
        use_fp16=use_fp16,
    )


def batch(
    detector_batch_size: Optional[int] = None,
    pose_estimator_batch_size: Optional[int] = None,
):
    detector_model = DetectorModel.rtmdet_medium()
    pose_model = PoseModel.rtmpose_large_384()

    honeycomb_client = HoneycombCachingClient()

    #############################################################################
    # Prepare the batch data (the time segments that poses should be generated for)
    #############################################################################
    batch_data = [
        ["dahlia", "2023-06-30 20:32:46+00:00", "2023-06-30 20:34:58+00:00"],
        ["dahlia", "2023-06-30 20:35:14+00:00", "2023-06-30 20:43:45+00:00"],
        ["dahlia", "2023-06-30 20:44:59+00:00", "2023-06-30 21:34:00+00:00"],
        ["dahlia", "2023-06-30 21:37:54+00:00", "2023-06-30 21:47:05+00:00"],
        # ["dahlia", "2023-07-20 19:31:48+00:00", "2023-07-20 19:33:14+00:00"],
        # ["dahlia", "2023-07-20 19:33:33+00:00", "2023-07-20 19:42:52+00:00"],
        # ["dahlia", "2023-07-20 19:43:25+00:00", "2023-07-20 19:47:25+00:00"],
        # ["dahlia", "2023-07-20 19:58:09+00:00", "2023-07-20 20:02:46+00:00"],
        # ["dahlia", "2023-07-20 20:25:10+00:00", "2023-07-20 20:28:49+00:00"],
        # ["dahlia", "2023-07-20 20:46:29+00:00", "2023-07-20 21:03:02+00:00"],
        # ["dahlia", "2023-07-20 21:05:21+00:00", "2023-07-20 21:09:50+00:00"],
        # ["dahlia", "2023-07-20 21:10:33+00:00", "2023-07-20 21:35:21+00:00"],
        # ["dahlia", "2023-07-20 21:43:39+00:00", "2023-07-20 21:53:20+00:00"],
        # ["dahlia", "2023-07-20 21:59:33+00:00", "2023-07-20 22:07:21+00:00"],
        # ["dahlia", "2023-07-20 22:27:51+00:00", "2023-07-20 22:39:33+00:00"],
        # ["dahlia", "2023-07-20 22:42:25+00:00", "2023-07-20 22:43:58+00:00"],
        # ["dahlia", "2023-07-20 22:57:15+00:00", "2023-07-20 23:07:14+00:00"],
        # ["dahlia", "2023-07-20 23:12:27+00:00", "2023-07-20 23:17:14+00:00"],
        # ["dahlia", "2023-07-20 23:17:15+00:00", "2023-07-20 23:17:40+00:00"],
        # ["dahlia", "2023-07-20 23:18:03+00:00", "2023-07-20 23:25:51+00:00"],
    ]
    batch_data = list(
        map(
            lambda v: [
                v[0],
                datetime.fromisoformat(v[1]),
                datetime.fromisoformat(v[2]),
            ],
            batch_data,
        )
    )
    df_batch = pd.DataFrame(
        batch_data,
        columns=["environment_name", "start", "end"],
    )
    df_batch["start"] = df_batch["start"] - timedelta(seconds=10)
    df_batch["end"] = df_batch["end"] + timedelta(seconds=10)

    #############################################################################
    # Load information about the existing poses, this is to avoid generating poses more than once
    #############################################################################
    unique_environments = df_batch["environment_name"].unique()

    all_existing_inferences = []
    check_against_existing_coverage = False
    if check_against_existing_coverage:
        for environment in unique_environments:
            environment_id = honeycomb_client.fetch_environment(environment)[
                "environment_id"
            ]

            df_existing_inferences = (
                PoseHandle().fetch_pose_2d_coverage_dataframe_by_environment_id(
                    environment_id=environment_id
                )
            )
            all_existing_inferences.append(df_existing_inferences)

    if len(all_existing_inferences) == 0:
        df_existing_inferences = pd.DataFrame(
            [], columns=["environment_id", "environment_name", "start", "end"]
        )
    else:
        df_existing_inferences = pd.concat(all_existing_inferences)

        df_existing_inferences["environment_name"] = df_existing_inferences[
            "environment_id"
        ].apply(
            lambda e_id: honeycomb_client.fetch_environment(environment=e_id)[
                "environment_name"
            ]
        )

    def merge_batch(_df_batch):
        """
        Merge overlapping start-end times
        """

        segments = []
        for _, segment in _df_batch.iterrows():
            environment = segment["environment_name"]
            start = segment["start"]
            end = segment["end"]
            if len(segments) == 0:
                segments.append(
                    {
                        "environment_name": environment,
                        "start": start,
                        "end": end,
                    }
                )
                continue

            merged = False
            for s in segments:
                if environment != s["environment_name"]:
                    continue

                if start < s["start"] and s["start"] <= end <= s["end"]:
                    s["start"] = start
                    merged = True
                    break
                if end > s["end"] and s["start"] <= start <= s["end"]:
                    s["end"] = end
                    merged = True
                    break
                if start >= s["start"] and end <= s["end"]:
                    merged = True
                    break

            if not merged:
                segments.append(
                    {
                        "environment_name": environment,
                        "start": start,
                        "end": end,
                    }
                )

        if len(segments) > 0:
            df_merged_segments = pd.DataFrame(segments).sort_values(by=["start"])
        else:
            df_merged_segments = pd.DataFrame([], columns=_df_batch.columns)

        return df_merged_segments

    df_existing_merged_segments = merge_batch(df_existing_inferences)
    df_batch_merged_segments = merge_batch(df_batch)

    #############################################################################
    # Remove time segments that have already been processed by the pose_engine
    #############################################################################
    segments = []
    for (
        environment_name,
        df_batch_merged_segments_by_environment,
    ) in df_batch_merged_segments.groupby(by="environment_name"):
        df_existing_merged_segments_by_environment = df_existing_merged_segments[
            df_existing_merged_segments["environment_name"] == environment_name
        ]

        existing_time_set = set()
        for _, row in df_existing_merged_segments_by_environment.iterrows():
            set_date_range = set(
                pd.date_range(start=row["start"], end=row["end"], freq="100L")
            )
            existing_time_set = existing_time_set.union(set_date_range)

        new_time_set = set()
        for _, row in df_batch_merged_segments_by_environment.iterrows():
            set_date_range = set(
                pd.date_range(start=row["start"], end=row["end"], freq="100L")
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
                    "environment_name": environment_name,
                    "start": df_valid_group["frame_time"].min(),
                    "end": df_valid_group["frame_time"].max(),
                }
            )
    df_merged_segments = pd.DataFrame(segments)

    #############################################################################
    # Process poses by looping through every time segment in df_merged_segments
    #############################################################################
    process_manager = mp.Manager()

    p = Pipeline(
        mp_manager=process_manager,
        detector_model=detector_model,
        detector_device="cuda:0",
        pose_estimator_model=pose_model,
        pose_estimator_device="cuda:1",
        use_fp16=False,
        detector_batch_size=detector_batch_size,
        pose_estimator_batch_size=pose_estimator_batch_size,
    )

    df_merged_segments["length"] = (
        df_merged_segments["end"] - df_merged_segments["start"]
    )
    logger.info(
        f"Starting pose processing, working through {df_merged_segments['length'].sum()} of the classroom"
    )
    for ii, (_, b) in enumerate(df_merged_segments.iterrows()):
        logger.info(f"Processing classroom segment {ii+1}/{len(df_merged_segments)}")
        environment = b["environment_name"]
        start = b["start"].to_pydatetime()
        end = b["end"].to_pydatetime()

        p.run(environment=environment, start_datetime=start, end_datetime=end)
