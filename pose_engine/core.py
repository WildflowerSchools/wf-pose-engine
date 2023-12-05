from datetime import datetime, timedelta

import pandas as pd
import torch.multiprocessing as mp

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

    old_batch = [
        # RUN ONE
        ["dahlia", "2023-10-02 15:32:38+00:00", "2023-10-02 15:33:14.400000+00:00"],
        [
            "dahlia",
            "2023-10-02 15:34:32.300000+00:00",
            "2023-10-02 15:34:56.800000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 15:35:10.300000+00:00",
            "2023-10-02 15:35:14.800000+00:00",
        ],
        ["dahlia", "2023-10-02 15:35:20+00:00", "2023-10-02 15:35:28+00:00"],
        [
            "dahlia",
            "2023-10-02 15:37:03.600000+00:00",
            "2023-10-02 15:37:37.100000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 15:47:26.600000+00:00",
            "2023-10-02 15:47:35.100000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 15:54:55.900000+00:00",
            "2023-10-02 15:55:40.500000+00:00",
        ],
        ["dahlia", "2023-10-02 15:56:58+00:00", "2023-10-02 15:57:06.100000+00:00"],
        [
            "dahlia",
            "2023-10-02 15:57:59.100000+00:00",
            "2023-10-02 15:59:34.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:01:48.400000+00:00",
            "2023-10-02 16:03:06.800000+00:00",
        ],
        ["dahlia", "2023-10-02 16:05:57+00:00", "2023-10-02 16:10:58.500000+00:00"],
        [
            "dahlia",
            "2023-10-02 16:11:45.200000+00:00",
            "2023-10-02 16:11:47.800000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:14:14.300000+00:00",
            "2023-10-02 16:15:58.400000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:19:01.500000+00:00",
            "2023-10-02 16:20:03.600000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:20:20.900000+00:00",
            "2023-10-02 16:20:43.400000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:22:00.500000+00:00",
            "2023-10-02 16:27:00.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:38:22.300000+00:00",
            "2023-10-02 16:39:20.200000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:39:52.900000+00:00",
            "2023-10-02 16:39:56.400000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:40:07.800000+00:00",
            "2023-10-02 16:42:21.900000+00:00",
        ],
        ["dahlia", "2023-10-02 16:42:22+00:00", "2023-10-02 16:46:05.900000+00:00"],
        [
            "dahlia",
            "2023-10-02 16:46:43.500000+00:00",
            "2023-10-02 16:49:21.100000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 16:56:27.700000+00:00",
            "2023-10-02 16:56:43.600000+00:00",
        ],
        # RUN TWO
        [
            "dahlia",
            "2023-10-02 17:04:30.500000+00:00",
            "2023-10-02 17:06:09.100000+00:00",
        ],
        ["dahlia", "2023-10-02 17:18:35+00:00", "2023-10-02 17:21:06+00:00"],
        [
            "dahlia",
            "2023-10-02 17:41:40.600000+00:00",
            "2023-10-02 17:41:46.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 17:42:03.900000+00:00",
            "2023-10-02 17:42:48.800000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 17:43:08.700000+00:00",
            "2023-10-02 17:43:15.900000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 17:43:33.200000+00:00",
            "2023-10-02 17:43:40.200000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 17:45:45.100000+00:00",
            "2023-10-02 17:50:48.600000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 17:51:10.700000+00:00",
            "2023-10-02 17:51:44.900000+00:00",
        ],
        ["dahlia", "2023-10-02 17:52:08+00:00", "2023-10-02 17:55:50.100000+00:00"],
        [
            "dahlia",
            "2023-10-02 17:56:02.500000+00:00",
            "2023-10-02 17:56:57.400000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 17:57:21.900000+00:00",
            "2023-10-02 17:57:45.900000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 17:59:37.300000+00:00",
            "2023-10-02 18:00:04.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 18:55:02.500000+00:00",
            "2023-10-02 18:58:31.400000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 18:59:41.500000+00:00",
            "2023-10-02 19:01:46.300000+00:00",
        ],
        ["dahlia", "2023-10-02 19:02:07+00:00", "2023-10-02 19:09:05.900000+00:00"],
        [
            "dahlia",
            "2023-10-02 19:10:17.500000+00:00",
            "2023-10-02 19:19:40.300000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 19:24:03.200000+00:00",
            "2023-10-02 19:24:38.900000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 19:25:39.800000+00:00",
            "2023-10-02 19:25:49.100000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 19:26:34.100000+00:00",
            "2023-10-02 19:26:53.900000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:05:51.500000+00:00",
            "2023-10-02 20:10:51.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:11:56.100000+00:00",
            "2023-10-02 20:13:29.300000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:13:30.500000+00:00",
            "2023-10-02 20:13:44.800000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:15:51.500000+00:00",
            "2023-10-02 20:20:56.700000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:23:08.600000+00:00",
            "2023-10-02 20:24:19.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:25:25.900000+00:00",
            "2023-10-02 20:26:48.200000+00:00",
        ],
        ["dahlia", "2023-10-02 20:27:03+00:00", "2023-10-02 20:28:09.800000+00:00"],
        [
            "dahlia",
            "2023-10-02 20:29:56.700000+00:00",
            "2023-10-02 20:30:07.200000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:30:21.800000+00:00",
            "2023-10-02 20:32:21.200000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:49:45.800000+00:00",
            "2023-10-02 20:51:14.200000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 21:05:24.500000+00:00",
            "2023-10-02 21:15:24.400000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 21:18:16.600000+00:00",
            "2023-10-02 21:21:30.400000+00:00",
        ],
        # RUN THREE
        [
            "dahlia",
            "2023-10-02 21:28:57.700000+00:00",
            "2023-10-02 21:29:09.700000+00:00",
        ],
        # RUN FOUR
        [
            "dahlia",
            "2023-10-02 21:29:30.300000+00:00",
            "2023-10-02 21:30:41.200000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 22:22:12.600000+00:00",
            "2023-10-02 22:22:15.700000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 23:07:19.100000+00:00",
            "2023-10-02 23:07:36.800000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 23:14:49.500000+00:00",
            "2023-10-02 23:14:56.100000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 23:16:11.200000+00:00",
            "2023-10-02 23:16:16.600000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 23:16:39.500000+00:00",
            "2023-10-02 23:16:44.300000+00:00",
        ],
        # Coverage from RUN ONE w/ No Interactions
        ["dahlia", "2023-10-02 15:13:46+00:00", "2023-10-02 15:16:41.500000+00:00"],
        ["dahlia", "2023-10-02 15:31:05.500000+00:00", "2023-10-02 16:20:48+00:00"],
        ["dahlia", "2023-10-02 16:22:00.500000+00:00", "2023-10-02 16:31:14+00:00"],
        ["dahlia", "2023-10-02 16:32:22+00:00", "2023-10-02 16:46:05.900000+00:00"],
        ["dahlia", "2023-10-02 16:46:43.500000+00:00", "2023-10-02 16:49:22+00:00"],
        ["dahlia", "2023-10-02 16:49:55.500000+00:00", "2023-10-02 17:07:56+00:00"],
        ["dahlia", "2023-10-02 17:09:16+00:00", "2023-10-02 17:14:41.500000+00:00"],
        ["dahlia", "2023-10-02 17:15:30.500000+00:00", "2023-10-02 18:01:14+00:00"],
        ["dahlia", "2023-10-02 18:07:51.500000+00:00", "2023-10-02 18:10:52+00:00"],
        [
            "dahlia",
            "2023-10-02 18:12:27.500000+00:00",
            "2023-10-02 18:15:10.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 18:26:07.500000+00:00",
            "2023-10-02 18:29:53.500000+00:00",
        ],
        ["dahlia", "2023-10-02 18:36:55+00:00", "2023-10-02 18:40:02.500000+00:00"],
        ["dahlia", "2023-10-02 18:52:07+00:00", "2023-10-02 18:58:31.400000+00:00"],
        ["dahlia", "2023-10-02 18:58:55.500000+00:00", "2023-10-02 19:22:21+00:00"],
    ]
    old_batch = list(
        map(
            lambda v: [
                v[0],
                datetime.fromisoformat(v[1]),
                datetime.fromisoformat(v[2]),
            ],
            old_batch,
        )
    )
    # df_merged_segments = pd.DataFrame(old_batch, columns=["environment", "start_datetime", "end_datetime"])
    # df_merged_segments['start_datetime'] = df_merged_segments['start_datetime'] - timedelta(seconds=10)
    # df_merged_segments['end_datetime'] = df_merged_segments['end_datetime'] + timedelta(seconds=10)

    batch_with_no_interaction = [
        # 2023-10-02
        ["dahlia", "2023-10-02 15:13:46+00:00", "2023-10-02 15:16:41.500000+00:00"],
        ["dahlia", "2023-10-02 15:31:05.500000+00:00", "2023-10-02 16:20:48+00:00"],
        ["dahlia", "2023-10-02 16:22:00.500000+00:00", "2023-10-02 16:31:14+00:00"],
        ["dahlia", "2023-10-02 16:32:22+00:00", "2023-10-02 16:46:05.900000+00:00"],
        ["dahlia", "2023-10-02 16:46:43.500000+00:00", "2023-10-02 16:49:22+00:00"],
        ["dahlia", "2023-10-02 16:49:55.500000+00:00", "2023-10-02 17:07:56+00:00"],
        ["dahlia", "2023-10-02 17:09:16+00:00", "2023-10-02 17:14:41.500000+00:00"],
        ["dahlia", "2023-10-02 17:15:30.500000+00:00", "2023-10-02 18:01:14+00:00"],
        ["dahlia", "2023-10-02 18:07:51.500000+00:00", "2023-10-02 18:10:52+00:00"],
        [
            "dahlia",
            "2023-10-02 18:12:27.500000+00:00",
            "2023-10-02 18:15:10.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 18:26:07.500000+00:00",
            "2023-10-02 18:29:53.500000+00:00",
        ],
        ["dahlia", "2023-10-02 18:36:55+00:00", "2023-10-02 18:40:02.500000+00:00"],
        ["dahlia", "2023-10-02 18:52:07+00:00", "2023-10-02 18:58:31.400000+00:00"],
        ["dahlia", "2023-10-02 18:58:55.500000+00:00", "2023-10-02 19:22:21+00:00"],
        ["dahlia", "2023-10-02 19:23:04+00:00", "2023-10-02 19:28:09.500000+00:00"],
        ["dahlia", "2023-10-02 19:35:55+00:00", "2023-10-02 19:38:38.500000+00:00"],
        ["dahlia", "2023-10-02 19:39:14+00:00", "2023-10-02 19:43:42.500000+00:00"],
        [
            "dahlia",
            "2023-10-02 19:49:23.500000+00:00",
            "2023-10-02 19:55:35.500000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 20:05:51.500000+00:00",
            "2023-10-02 20:35:51.500000+00:00",
        ],
        ["dahlia", "2023-10-02 20:44:14+00:00", "2023-10-02 20:53:06.500000+00:00"],
        ["dahlia", "2023-10-02 20:54:26.500000+00:00", "2023-10-02 20:59:41+00:00"],
        [
            "dahlia",
            "2023-10-02 21:05:24.500000+00:00",
            "2023-10-02 21:15:24.400000+00:00",
        ],
        [
            "dahlia",
            "2023-10-02 21:15:24.500000+00:00",
            "2023-10-02 21:21:30.500000+00:00",
        ],
        ["dahlia", "2023-10-02 21:23:47+00:00", "2023-10-02 21:30:43+00:00"],
        [
            "dahlia",
            "2023-10-02 22:20:02.500000+00:00",
            "2023-10-02 22:23:37.500000+00:00",
        ],
        ["dahlia", "2023-10-02 22:45:55.500000+00:00", "2023-10-02 22:48:01+00:00"],
        ["dahlia", "2023-10-02 22:56:15+00:00", "2023-10-02 22:58:08.500000+00:00"],
        ["dahlia", "2023-10-02 22:58:54.500000+00:00", "2023-10-02 23:01:07+00:00"],
        ["dahlia", "2023-10-02 23:06:11.500000+00:00", "2023-10-02 23:16:51+00:00"]
        # 2023-06-30
        # ["dahlia", "2023-06-30 20:32:46+00:00", "2023-06-30 20:34:58+00:00"],
        # ["dahlia", "2023-06-30 20:35:14+00:00", "2023-06-30 20:43:45+00:00"],
        # ["dahlia", "2023-06-30 20:44:59+00:00", "2023-06-30 21:34:00+00:00"],
        # ["dahlia", "2023-06-30 21:37:54+00:00", "2023-06-30 21:47:05+00:00"]
    ]
    batch_with_no_interaction = list(
        map(
            lambda v: [
                v[0],
                datetime.fromisoformat(v[1]),
                datetime.fromisoformat(v[2]),
            ],
            batch_with_no_interaction,
        )
    )

    df_old_batch = pd.DataFrame(
        old_batch, columns=["environment", "start_datetime", "end_datetime"]
    )
    df_old_batch["start_datetime"] = df_old_batch["start_datetime"] - timedelta(
        seconds=10
    )
    df_old_batch["end_datetime"] = df_old_batch["end_datetime"] + timedelta(seconds=10)

    df_batch_with_no_interaction = pd.DataFrame(
        batch_with_no_interaction,
        columns=["environment", "start_datetime", "end_datetime"],
    )
    df_batch_with_no_interaction["start_datetime"] = df_batch_with_no_interaction[
        "start_datetime"
    ] - timedelta(seconds=10)
    df_batch_with_no_interaction["end_datetime"] = df_batch_with_no_interaction[
        "end_datetime"
    ] + timedelta(seconds=10)

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

    df_old_batch_merged_segments = merge_batch(df_old_batch)
    df_batch_with_no_interaction_merged_segments = merge_batch(
        df_batch_with_no_interaction
    )

    old_batch_time_set = set()
    for _, row in df_old_batch_merged_segments.iterrows():
        set_date_range = set(
            pd.date_range(
                start=row["start_datetime"], end=row["end_datetime"], freq="100L"
            )
        )
        old_batch_time_set = old_batch_time_set.union(set_date_range)

    no_interaction_time_set = set()
    for _, row in df_batch_with_no_interaction_merged_segments.iterrows():
        set_date_range = set(
            pd.date_range(
                start=row["start_datetime"], end=row["end_datetime"], freq="100L"
            )
        )
        no_interaction_time_set = no_interaction_time_set.union(set_date_range)

    valid_times = no_interaction_time_set - old_batch_time_set
    df_valid_times = (
        pd.DataFrame(valid_times, columns=["frame_time"])
        .sort_values(by="frame_time")
        .reset_index(drop=True)
    )
    df_valid_times["gap"] = df_valid_times["frame_time"].diff() > pd.to_timedelta(
        "100L"
    )
    # df_valid_times.loc[df_valid_times.index[0], 'split'] = True

    df_valid_times["group"] = df_valid_times["gap"].cumsum()
    segments = []
    for _, df_valid_group in df_valid_times.groupby(by="group"):
        segments.append(
            {
                "environment": "dahlia",
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
