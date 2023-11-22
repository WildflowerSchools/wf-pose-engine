import torch.utils.data

from pose_engine.log import logger


class BoundingBoxesDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = self._collate_fn

    def _collate_fn(self, data):
        return tuple(zip(*data))

    def __del__(self):
        del self.dataset
