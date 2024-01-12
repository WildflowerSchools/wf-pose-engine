import time
from typing import Any, Dict, Union

import torch

from pose_engine.log import logger


class MMLabCompatibleDataParallel(torch.nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super(MMLabCompatibleDataParallel, self).__init__(*args, **kwargs)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Override the original forward function.
        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        if not self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
        else:
            return super().forward(*inputs, **kwargs)

    def test_step(self, data):
        self.module._run_forward = self._run_forward

        return self.module.test_step(data)

    def _run_forward(
        self, data: Union[dict, tuple, list], mode: str
    ) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data and passes it to the DataParallel replicate and parallel_apply methods. Uses a custom gather method.`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        start = time.time()

        if isinstance(data, (dict)):
            s = time.time()
            scattered_inputs, _ = self.scatter(
                [data["inputs"], data["data_samples"]],
                kwargs=None,
                device_ids=self.device_ids,
            )
            logger.info(f"Profile: scatter: {round(time.time() - s, 2)} seconds")

            def chunk_list(input, chunk_size):
                for i in range(0, len(input), chunk_size):
                    yield input[i : i + chunk_size]

            s = time.time()
            scattered_inputs_inputs = []
            scattered_kwargs_data_samples = []
            for idx, scattered_input in enumerate(scattered_inputs):
                inputs = scattered_input[0]
                data_samples = scattered_input[1]

                inputs_size = len(scattered_input[0])
                chunked_data_samples = list(chunk_list(data_samples, inputs_size))

                scattered_inputs_inputs.append(inputs)
                scattered_kwargs_data_samples.append(
                    {"data_samples": chunked_data_samples[idx], "mode": "predict"}
                )
            logger.info(
                f"Profile: scatter augment: {round(time.time() - s, 2)} seconds"
            )

            s = time.time()
            replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
            logger.info(f"Profile: replicate: {round(time.time() - s, 2)} seconds")
            s = time.time()
            outputs = self.parallel_apply(
                replicas, scattered_inputs_inputs, scattered_kwargs_data_samples
            )
            logger.info(
                f"Profile: parallel inference: {round(time.time() - s, 2)} seconds"
            )

            s = time.time()
            results = []
            for output in outputs:
                for pose_data_sample in output:
                    results.append(pose_data_sample.to(self.output_device))
            logger.info(f"Profile: output augment: {round(time.time() - s, 2)} seconds")
        else:
            raise TypeError(
                "Output of `data_preprocessor` should be " f"dict, but got {type(data)}"
            )

        logger.info(
            f"Profile: total dataparallel runtime: {round(time.time() - start, 2)} seconds"
        )
        return results
