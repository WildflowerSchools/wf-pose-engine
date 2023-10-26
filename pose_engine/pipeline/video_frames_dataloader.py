import torch.utils.data

from pose_engine.log import logger


class VideoFramesDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, device="cpu", **kwargs):
        super(VideoFramesDataLoader, self).__init__(*args, **kwargs)

        self.device = device

    def move_to_device(self, batch):
        frames, meta = batch
        return frames.to(device=self.device), meta

    def __iter__(self):
        for d in super(VideoFramesDataLoader, self).__iter__():
            yield (self.move_to_device(d))
