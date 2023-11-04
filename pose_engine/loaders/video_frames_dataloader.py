import torch.utils.data

from pose_engine.log import logger


class VideoFramesDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, device="cpu", **kwargs):
        super(VideoFramesDataLoader, self).__init__(*args, **kwargs)

        self.device = device

    def move_to_device(self, batch):
        frames, meta = batch

        meta_to_device = {}
        for key, value in meta.items():
            if isinstance(value, torch.Tensor):
                meta_to_device[key] = value.to(device=self.device)
            else:
                meta_to_device[key] = value

        return frames.to(device=self.device), meta_to_device

    def __iter__(self):
        logger.debug("Video frame dataloader is beginning its iteration...")
        for d in super(VideoFramesDataLoader, self).__iter__():
            yield (self.move_to_device(d))

        logger.debug("Video frame dataloader is done iterating")
