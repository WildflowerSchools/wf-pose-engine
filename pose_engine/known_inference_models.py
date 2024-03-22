from typing import Optional, Union

import torch

from pose_db_io.handle.models import pose_2d


class InferenceModel:
    def __init__(
        self,
        model_runtime: pose_2d.ModelRuntime,
        model_config: str,
        model_config_enum: Union[
            pose_2d.DetectorModelConfigEnum, pose_2d.PoseModelConfigEnum
        ],
        checkpoint: str,
        checkpoint_enum: Union[
            pose_2d.DetectorModelCheckpointEnum, pose_2d.PoseModelCheckpointEnum
        ],
        deployment_config: Optional[str] = None,
        deployment_config_enum: Optional[
            Union[
                pose_2d.DetectorModelDeploymentConfigEnum,
                pose_2d.PoseModelDeploymentConfigEnum,
            ]
        ] = None,
    ):
        self.model_runtime = model_runtime
        self.model_config = model_config
        self.model_config_enum = model_config_enum
        self.checkpoint = checkpoint
        self.checkpoint_enum = checkpoint_enum
        self.deployment_config = deployment_config
        self.deployment_config_enum = deployment_config_enum


class DetectorModel(InferenceModel):
    def __init__(
        self, *args, bounding_box_format_enum: pose_2d.BoundingBoxFormatEnum, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.bounding_box_format_enum = bounding_box_format_enum

    @staticmethod
    def rtmdet_nano():
        return DetectorModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/mmdet/rtmdet_nano_640-8xb32_coco-person.py",
            model_config_enum=pose_2d.DetectorModelConfigEnum.rtmdet_nano_640_8xb32_coco_person,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
            checkpoint_enum=pose_2d.DetectorModelCheckpointEnum.rtmdet_nano_8xb32_100e_coco_obj365_person_05d8511e,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
        )

    @staticmethod
    def rtmdet_medium():
        return DetectorModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
            model_config_enum=pose_2d.DetectorModelConfigEnum.rtmdet_m_640_8xb32_coco_person,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
            checkpoint_enum=pose_2d.DetectorModelCheckpointEnum.rtmdet_m_8xb32_100e_coco_obj365_person_235e8209,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
        )

    @staticmethod
    def rtmdet_medium_tensorrt_static_640x640():
        return DetectorModel(
            model_runtime=pose_2d.ModelRuntime.tensorrt,
            model_config="./configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
            model_config_enum=pose_2d.DetectorModelConfigEnum.rtmdet_m_640_8xb32_coco_person,
            checkpoint="./checkpoints/rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640.engine",
            checkpoint_enum=pose_2d.DetectorModelCheckpointEnum.rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640,
            deployment_config="./configs/tensorrt/mmdet/detection_tensorrt_static-640x640.py",
            deployment_config_enum=pose_2d.DetectorModelDeploymentConfigEnum.tensorrt_static_640x640,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
        )

    @staticmethod
    def rtmdet_medium_tensorrt_dynamic_640x640_batch():
        return DetectorModel(
            model_runtime=pose_2d.ModelRuntime.tensorrt,
            model_config="./configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
            model_config_enum=pose_2d.DetectorModelConfigEnum.rtmdet_m_640_8xb32_coco_person,
            checkpoint="./checkpoints/rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_batch.engine",
            checkpoint_enum=pose_2d.DetectorModelCheckpointEnum.rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_batch,
            deployment_config="./configs/tensorrt/mmdet/detection_tensorrt_dynamic-640x640_batch.py",
            deployment_config_enum=pose_2d.DetectorModelDeploymentConfigEnum.tensorrt_dynamic_640x640_batch,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
        )

    @staticmethod
    def rtmdet_medium_tensorrt_dynamic_640x640_fp16_batch():
        return DetectorModel(
            model_runtime=pose_2d.ModelRuntime.tensorrt,
            model_config="./configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
            model_config_enum=pose_2d.DetectorModelConfigEnum.rtmdet_m_640_8xb32_coco_person,
            checkpoint="./checkpoints/rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_fp16_batch.engine",
            checkpoint_enum=pose_2d.DetectorModelCheckpointEnum.rtmdet_m_8xb32_100e_coco_obj365_person_235e8209_tensorrt_static_640x640_fp16_batch,
            deployment_config="./configs/tensorrt/mmdet/detection_tensorrt_dynamic-640x640_fp16_batch.py",
            deployment_config_enum=pose_2d.DetectorModelDeploymentConfigEnum.tensorrt_dynamic_640x640_fp16_batch,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
        )


class PoseModel(InferenceModel):
    def __init__(
        self,
        *args,
        keypoint_format_enum: pose_2d.KeypointsFormatEnum,
        pose_estimator_type: pose_2d.PoseEstimatorType,
        bounding_box_format_enum: Optional[
            pose_2d.BoundingBoxFormatEnum
        ] = None,  # BoundingBoxFormatEnum is included here because of one-stage models
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.bounding_box_format_enum = bounding_box_format_enum
        self.keypoint_format_enum = keypoint_format_enum
        self.pose_estimator_type = pose_estimator_type

    @staticmethod
    def rtmpose_small_256():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb256-420e_body8-256x192.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmpose_s_8xb256_420e_body8_256x192,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_s_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.top_down,
        )

    @staticmethod
    def rtmpose_medium_256():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmpose_m_8xb256_420e_body8_256x192,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_m_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.top_down,
        )

    @staticmethod
    def rtmpose_medium_384():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-384x288.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmpose_m_8xb256_420e_body8_384x288,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_m_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.top_down,
        )

    @staticmethod
    def rtmpose_large_256():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmpose_l_8xb256_420e_body8_256x192,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_l_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.top_down,
        )

    @staticmethod
    def rtmpose_large_384():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-384x288.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmpose_l_8xb256_420e_body8_384x288,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.top_down,
        )

    @staticmethod
    def rtmpose_large_384_tensorrt_batch():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.tensorrt,
            model_config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-384x288.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmpose_l_8xb256_420e_body8_384x288,
            checkpoint="./checkpoints/rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504_tensorrt_dynamic_384x288_batch.engine",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504_tensorrt_dynamic_384x288_batch,
            deployment_config="./configs/tensorrt/body_2d_keypoint/pose-detection_simcc_tensorrt_dynamic-384x288_batch.py",
            deployment_config_enum=pose_2d.PoseModelDeploymentConfigEnum.tensorrt_simcc_dynamic_384x288_batch,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.top_down,
        )

    @staticmethod
    def rtmo_large():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmo_l_16xb16_600e_body7_640x640,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.one_stage,
        )

    @staticmethod
    def rtmo_large_onnx():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.onnx,
            model_config="./configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmo_l_16xb16_600e_body7_640x640,
            checkpoint="https://wildflower-tech-model-zoo.s3.us-east-2.amazonaws.com/mmlab/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211/end2end.onnx",
            # checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211_onnx,
            deployment_config="./configs/runtimes/mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic.py",
            deployment_config_enum=pose_2d.PoseModelDeploymentConfigEnum.pose_detection_rtmo_onnxruntime_dynamic,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.one_stage,
        )

    @staticmethod
    def rtmo_large_onnx_fp16():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.onnx,
            model_config="./configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmo_l_16xb16_600e_body7_640x640,
            checkpoint="https://wildflower-tech-model-zoo.s3.us-east-2.amazonaws.com/mmlab/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211-fp16/end2end.onnx",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211_onnx_fp16,
            deployment_config="./configs/runtimes/mmdeploy/configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic-fp16.py",
            deployment_config_enum=pose_2d.PoseModelDeploymentConfigEnum.pose_detection_rtmo_onnxruntime_dynamic_fp16,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.one_stage,
        )

    @staticmethod
    def rtmo_large_tensorrt_fp16():
        if torch.cuda.device_count() == 0:
            raise ValueError("CUDA device must be present to run tensorrt model")

        if torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 2080 SUPER":
            checkpoint = "https://wildflower-tech-model-zoo.s3.us-east-2.amazonaws.com/mmlab/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640-rtx2080/end2end.engine"
            checkpoint_enum = (
                pose_2d.PoseModelCheckpointEnum.rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211_tensorrt_fp16_rtx2080
            )
        elif torch.cuda.get_device_name(0) == "Tesla V100-SXM2-16GB":
            checkpoint = "./mmdeploy_model/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640-rtx2080/end2end.engine"
            checkpoint_enum = (
                pose_2d.PoseModelCheckpointEnum.rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211_tensorrt_fp16_v100
            )
        elif torch.cuda.get_device_name(0) == "NVIDIA A10G":
            checkpoint = "https://wildflower-tech-model-zoo.s3.us-east-2.amazonaws.com/mmlab/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640-a10g/end2end.engine"
            checkpoint_enum = (
                pose_2d.PoseModelCheckpointEnum.rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211_tensorrt_fp16_a10g
            )
        elif torch.cuda.get_device_name(0) == "Tesla T4":
            checkpoint = "https://wildflower-tech-model-zoo.s3.us-east-2.amazonaws.com/mmlab/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640-t40/end2end.engine"
            checkpoint_enum = (
                pose_2d.PoseModelCheckpointEnum.rtmo_l_16xb16_600e_body7_640x640_b37118ce_20231211_tensorrt_fp16_t4
            )
        else:
            raise ValueError(
                f"RTMO tensorrt model not compiled for {torch.cuda.get_device_properties(0)}"
            )

        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.tensorrt,
            model_config="./configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmo_l_16xb16_600e_body7_640x640,
            checkpoint=checkpoint,
            checkpoint_enum=checkpoint_enum,
            deployment_config="./configs/runtimes/mmdeploy/configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py",
            deployment_config_enum=pose_2d.PoseModelDeploymentConfigEnum.pose_detection_rtmo_tensorrt_fp16_dynamic_640x640,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.one_stage,
        )

    @staticmethod
    def rtmo_medium():
        return PoseModel(
            model_runtime=pose_2d.ModelRuntime.pytorch,
            model_config="./configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py",
            model_config_enum=pose_2d.PoseModelConfigEnum.rtmo_m_16xb16_600e_body7_640x640,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmo_m_16xb16_600e_body7_640x640_39e78cc4_20231211,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
            pose_estimator_type=pose_2d.PoseEstimatorType.one_stage,
        )
