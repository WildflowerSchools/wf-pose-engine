from typing import Union

import torch
from torch import nn


class MMLabCompatibleDataParallel(torch.nn.DataParallel):
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        return self.module.test_step(data)
