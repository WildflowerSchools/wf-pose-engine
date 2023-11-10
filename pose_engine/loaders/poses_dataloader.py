import torch.utils.data


class PosesDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = self._collate_fn

    def _collate_fn(self, data):
        return tuple(zip(*data))
