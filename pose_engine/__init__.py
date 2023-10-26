from .log import logger

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError) as e:
    logger.exception(e)
    raise e
