import torch.utils.data

from pose_engine.log import logger
from pose_engine.pipeline.video_frames_dataset import VideoFramesDataset


def test_dataset_batch_size_1():
    dataset = VideoFramesDataset(
        video_paths=[
            "./input/test_video/output000.mp4",
            "./input/test_video/output001.mp4",
            "./input/test_video/output002.mp4",
            "./input/test_video/output003.mp4",
            "./input/test_video/output004.mp4",
        ],
        wait_for_video_files=False)
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)

    processed_batches = 0
    for _, data in enumerate(loader):
        processed_batches += 1
        if data is None:
            break
    
    assert processed_batches == 500, "Expected VideoFramesDataset to generate 500 video batches, instead it generated {processed_batches} batches"
    dataset.stop_video_loader()


def test_dataset_batch_size_2():
    dataset = VideoFramesDataset(
        video_paths=[
            "./input/test_video/output000.mp4",
            "./input/test_video/output001.mp4",
        ],
        wait_for_video_files=False)
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0, batch_size=2)

    processed_batches = 0
    for _, data in enumerate(loader):
        processed_batches += 1
        if data is None:
            break
        
        assert len(data) == 2, f"Expected batched frames of size 2, instead received batch of size {len(data)}"
    
    assert processed_batches == 100, f"Expected VideoFramesDataset to generate 100 video frame batches, instead it generated {processed_batches} batches"
    dataset.stop_video_loader()
