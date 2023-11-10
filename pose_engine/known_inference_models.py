from .handle.models import pose_2d


class InferenceModel:
    def __init__(self, config, config_enum, checkpoint, checkpoint_enum):
        self.config = config
        self.config_enum = config_enum
        self.checkpoint = checkpoint
        self.checkpoint_enum = checkpoint_enum


class DetectorModel(InferenceModel):
    def __init__(self, *args, bounding_box_format_enum, **kwargs):
        super(DetectorModel, self).__init__(*args, **kwargs)

        self.bounding_box_format_enum = bounding_box_format_enum

    @staticmethod
    def rtmdet_nano():
        return DetectorModel(
            config="./configs/mmdet/rtmdet_nano_640-8xb32_coco-person.py",
            config_enum=pose_2d.DetectorModelConfigEnum.rtmdet_nano_640_8xb32_coco_person,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
            checkpoint_enum=pose_2d.DetectorModelCheckpointEnum.rtmdet_nano_8xb32_100e_coco_obj365_person_05d8511e,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
        )

    @staticmethod
    def rtmdet_medium():
        return DetectorModel(
            config="./configs/mmdet/rtmdet_m_640-8xb32_coco-person.py",
            config_enum=pose_2d.DetectorModelConfigEnum.rtmdet_m_640_8xb32_coco_person,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
            checkpoint_enum=pose_2d.DetectorModelCheckpointEnum.rtmdet_m_8xb32_100e_coco_obj365_person_235e8209,
            bounding_box_format_enum=pose_2d.BoundingBoxFormatEnum.xyxy,
        )


class PoseModel(InferenceModel):
    def __init__(self, *args, keypoint_format_enum, **kwargs):
        super(PoseModel, self).__init__(*args, **kwargs)

        self.keypoint_format_enum = keypoint_format_enum

    @staticmethod
    def rtmpose_small_256():
        return PoseModel(
            config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb256-420e_body8-256x192.py",
            config_enum=pose_2d.PoseModelConfigEnum.rtmpose_s_8xb256_420e_body8_256x192,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_s_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
        )

    @staticmethod
    def rtmpose_medium_256():
        return PoseModel(
            config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py",
            config_enum=pose_2d.PoseModelConfigEnum.rtmpose_m_8xb256_420e_body8_256x192,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_m_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
        )

    @staticmethod
    def rtmpose_medium_384():
        return PoseModel(
            config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-384x288.py",
            config_enum=pose_2d.PoseModelConfigEnum.rtmpose_m_8xb256_420e_body8_384x288,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_m_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
        )

    @staticmethod
    def rtmpose_large_256():
        return PoseModel(
            config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py",
            config_enum=pose_2d.PoseModelConfigEnum.rtmpose_l_8xb256_420e_body8_256x192,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_l_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
        )

    @staticmethod
    def rtmpose_large_384():
        return PoseModel(
            config="./configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-384x288.py",
            config_enum=pose_2d.PoseModelConfigEnum.rtmpose_l_8xb256_420e_body8_384x288,
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth",
            checkpoint_enum=pose_2d.PoseModelCheckpointEnum.rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504,
            keypoint_format_enum=pose_2d.KeypointsFormatEnum.coco_17,
        )
