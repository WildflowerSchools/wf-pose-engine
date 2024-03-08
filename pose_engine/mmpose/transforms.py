from typing import Dict, List, Optional, Tuple

import torch
import torchvision

import numpy as np

from mmpose.utils.typing import PoseDataSample


class BatchBottomupResize:
    def __init__(
        self,
        input_size: Tuple[int, int],
        aug_scales: Optional[List[float]] = None,
        size_factor: int = 32,
        resize_mode: str = "fit",
        pad_val: tuple = (0, 0, 0),
        use_udp: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.aug_scales = aug_scales
        self.resize_mode = resize_mode
        self.size_factor = size_factor
        self.use_udp = use_udp
        self.pad_val = pad_val

    @staticmethod
    def _ceil_to_multiple(size: Tuple[int, int], base: int):
        """Ceil the given size (tuple of [w, h]) to a multiple of the base."""
        return tuple(int(np.ceil(s / base) * base) for s in size)

    def _get_input_size(
        self, img_size: Tuple[int, int], input_size: Tuple[int, int]
    ) -> Tuple:
        """Calculate the actual input size (which the original image will be
        resized to) and the padded input size (which the resized image will be
        padded to, or which is the size of the model input).

        Args:
            img_size (Tuple[int, int]): The original image size in [w, h]
            input_size (Tuple[int, int]): The expected input size in [w, h]

        Returns:
            tuple:
            - actual_input_size (Tuple[int, int]): The target size to resize
                the image
            - padded_input_size (Tuple[int, int]): The target size to generate
                the model input which will contain the resized image
        """
        img_w, img_h = img_size
        ratio = img_w / img_h

        if self.resize_mode == "fit":
            padded_input_size = self._ceil_to_multiple(input_size, self.size_factor)
            if padded_input_size != input_size:
                raise ValueError(
                    "When ``resize_mode=='fit', the input size (height and"
                    " width) should be mulitples of the size_factor("
                    f"{self.size_factor}) at all scales. Got invalid input "
                    f"size {input_size}."
                )

            pad_w, pad_h = padded_input_size
            rsz_w = min(pad_w, pad_h * ratio)
            rsz_h = min(pad_h, pad_w / ratio)
            actual_input_size = (rsz_w, rsz_h)

        elif self.resize_mode == "expand":
            _padded_input_size = self._ceil_to_multiple(input_size, self.size_factor)
            pad_w, pad_h = _padded_input_size
            rsz_w = max(pad_w, pad_h * ratio)
            rsz_h = max(pad_h, pad_w / ratio)

            actual_input_size = (rsz_w, rsz_h)
            padded_input_size = self._ceil_to_multiple(
                actual_input_size, self.size_factor
            )

        else:
            raise ValueError(f"Invalid resize mode {self.resize_mode}")

        return actual_input_size, padded_input_size

    def transform(
        self, data_list: List[Dict], device: Optional[str] = None
    ) -> Optional[dict]:
        """The transform function of :class:`BottomupResize` to perform
        photometric distortion on images.

        See ``transform()`` method of :class:`BaseTransform` for details.


        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if len(data_list) == 0:
            return []

        if device is None:
            device = "cpu"

        imgs = (
            torch.stack(list(map(lambda r: r["inputs"], data_list)))
            .to(
                memory_format=torch.channels_last
            )  # Pytorch recommends using this option, but it doesn't appear to speed things up
            .to(device)
        )
        single_data_sample: PoseDataSample = data_list[0]["data_samples"]

        img_h, img_w = single_data_sample.get("ori_shape")
        w, h = self.input_size

        input_sizes = [(w, h)]
        if self.aug_scales:
            input_sizes += [(int(w * s), int(h * s)) for s in self.aug_scales]

        imgs_for_input_sizes = []
        for ii, (_w, _h) in enumerate(input_sizes):

            actual_input_size, padded_input_size = self._get_input_size(
                img_size=(img_w, img_h), input_size=(_w, _h)
            )

            center = np.array([img_w / 2, img_h / 2], dtype=np.float32)
            scale = np.array(
                [
                    img_w * padded_input_size[0] / actual_input_size[0],
                    img_h * padded_input_size[1] / actual_input_size[1],
                ],
                dtype=np.float32,
            )

            imgs_for_input_sizes.append(
                torchvision.transforms.Resize(padded_input_size)(imgs)
            )

            if ii == 0:
                # Store the transform information w.r.t. the main input size
                for data_list_item in data_list:
                    data_list_item["data_samples"].set_metainfo(
                        {
                            "img_shape": padded_input_size[::-1],
                            "input_center": center,
                            "input_scale": scale,
                            "input_size": padded_input_size,
                        }
                    )

        for ii, data_list_item in enumerate(data_list):
            if self.aug_scales:
                all_data_list_item_images = []
                for imgs_for_input_sizes_item in imgs_for_input_sizes:
                    all_data_list_item_images.append(imgs_for_input_sizes_item[ii])
                data_list_item["inputs"] = all_data_list_item_images
                # results['img'] = imgs
                data_list_item["data_samples"].set_metainfo(
                    dict(aug_scale=self.aug_scales)
                )
            else:
                data_list_item["inputs"] = imgs_for_input_sizes[0][ii]
                data_list_item["data_samples"].set_metainfo(dict(aug_scale=None))

        return data_list
