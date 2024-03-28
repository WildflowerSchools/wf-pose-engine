import torch
import torch.utils.data

# from pose_engine.log import logger


class VideoFramesDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

    def total_video_files_queued(self) -> int:
        return self.dataset.total_video_files_queued()

    def total_video_frames_queued(self) -> int:
        return self.dataset.total_video_frames_queued()

    # def move_to_device(self, batch):
    #     frames, meta = batch

    #     if torch.device(self.device) == frames.device:
    #         return frames, meta

    #     meta_to_device = {}
    #     for key, value in meta.items():
    #         if isinstance(value, torch.Tensor):
    #             meta_to_device[key] = value.to(device=self.device)
    #         else:
    #             meta_to_device[key] = value

    #     return frames.to(device=self.device), meta_to_device

    # def __iter__(self):
    #     logger.info("Video frame dataloader is beginning its iteration...")
    #     for d in super().__iter__():
    #         # for d in self.dataset.__iter__():
    #         logger.info(f"Video frame dataloader is yielding {len(d[0])} frames")
    #         yield self.move_to_device(d)
    #         logger.info(f"Video frame dataloader is done yielding {len(d[0])} frames")

    #     logger.debug("Video frame dataloader is done iterating")

    def __del__(self):
        del self.dataset
