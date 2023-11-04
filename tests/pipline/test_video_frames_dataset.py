import torch.utils.data

from pose_engine.log import logger
from pose_engine.loaders.video_frames_dataloader import VideoFramesDataLoader
from pose_engine.loaders.video_frames_dataset import VideoFramesDataset


def test_dataset_batch_size_1():
    batch_size = 1
    expected_frames = 500

    dataset = VideoFramesDataset(
        video_paths=[
            "./input/test_video/output000.mp4",
            "./input/test_video/output001.mp4",
            "./input/test_video/output002.mp4",
            "./input/test_video/output003.mp4",
            "./input/test_video/output004.mp4",
        ],
        wait_for_video_files=False)
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0, batch_size=batch_size, pin_memory=True)

    processed_batches = 0
    for _, (frames, meta) in enumerate(loader):
        processed_batches += 1
        if frames is None:
            break
        assert frames.shape[0] == batch_size, f"Expected batched frames of size 1, instead received batch of size {frames.shape[0]}"
    
    assert processed_batches == (expected_frames / batch_size), f"Expected VideoFramesDataset to generate {(expected_frames / batch_size)} video batches, instead it generated {processed_batches} batches"
    dataset.stop_video_loader()


def test_dataset_batch_size_2():
    batch_size = 2
    expected_frames = 500

    dataset = VideoFramesDataset(
        video_paths=[
            "./input/test_video/output000.mp4",
            "./input/test_video/output001.mp4",
            "./input/test_video/output002.mp4",
            "./input/test_video/output003.mp4",
            "./input/test_video/output004.mp4",
        ],
        wait_for_video_files=False)
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0, batch_size=batch_size, pin_memory=True)

    processed_batches = 0
    for _, (frames, meta) in enumerate(loader):
        processed_batches += 1
        if frames is None:
            break
        assert frames.shape[0] == batch_size, f"Expected batched frames of size {batch_size}, instead received batch of size {frames.shape[0]}"
    
    assert processed_batches == (expected_frames / batch_size), f"Expected VideoFramesDataset to generate {(expected_frames / batch_size)} video frame batches, instead it generated {processed_batches} batches"
    dataset.stop_video_loader()

def test_dataset_move_to_device_batch_size_1():
    batch_size = 1
    expected_frames = 100

    dataset = VideoFramesDataset(
        video_paths=[
            "./input/test_video/output000.mp4",
        ],
        wait_for_video_files=False)
    
    loader = VideoFramesDataLoader(dataset, device='cuda:0', shuffle=False, num_workers=0, batch_size=batch_size)

    processed_batches = 0
    for _, (frames, meta) in enumerate(loader):
        processed_batches += 1
        if frames is None:
            break
        assert frames.shape[0] == batch_size, f"Expected batched frames of size 1, instead received batch of size {frames.shape[0]}"
        assert frames.is_cuda, "Video frame tensors should be on CUDA:0 device"
    
    assert processed_batches == (expected_frames / batch_size), f"Expected VideoFramesDataset to generate {(expected_frames / batch_size)} video batches, instead it generated {processed_batches} batches"
    dataset.stop_video_loader()

def test_dataset_move_to_device_batch_size_4():
    batch_size = 4
    expected_frames = 100

    dataset = VideoFramesDataset(
        video_paths=[
            "./input/test_video/output000.mp4",
        ],
        wait_for_video_files=False)
    
    loader = VideoFramesDataLoader(dataset, device='cuda:0', shuffle=False, num_workers=0, batch_size=4)

    processed_batches = 0
    for _, (frames, meta) in enumerate(loader):
        processed_batches += 1
        if frames is None:
            break
        assert frames.shape[0] == batch_size, f"Expected batched frames of size 4, instead received batch of size {frames.shape[0]}"
        assert frames.is_cuda, "Video frame tensors should be on CUDA:0 device"
    
    assert processed_batches == (expected_frames / batch_size), f"Expected VideoFramesDataset to generate {(expected_frames / batch_size)} video batches, instead it generated {processed_batches} batches"
    dataset.stop_video_loader()