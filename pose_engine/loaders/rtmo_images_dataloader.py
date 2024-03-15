import torch.utils.data


class RTMOImagesDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = self._collate_fn

    def _collate_fn(self, data):
        return data

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

    def __iter__(self):
        for d in super().__iter__():
            # yield self.move_to_device(d)
            yield d

    def __del__(self):
        del self.dataset
