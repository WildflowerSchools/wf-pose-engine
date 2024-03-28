from typing import Optional, Sequence

from mmengine.runner import load_checkpoint
from mmdeploy.utils import Backend
from mmdeploy.backend.base import BACKEND_MANAGERS
from mmdeploy.backend.onnxruntime import ORTWrapper


@BACKEND_MANAGERS.register(Backend.ONNXRUNTIME.value)
class CustomORTWrapper(ORTWrapper):
    def __init__(
        self, onnx_file: str, device: str, output_names: Optional[Sequence[str]] = None
    ):

        checkpoint = load_checkpoint(onnx_file, map_location="cpu")
        super(CustomORTWrapper, self).__init__(
            onnx_file=onnx_file, device=device, output_names=output_names
        )
