from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from typing_extensions import Annotated
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer, UUID4
from pydantic.functional_validators import AfterValidator


class BoundingBoxFormatEnum(Enum):
    xyxy = "xyxy"
    xywh = "xywh"


class KeypointsFormatEnum(Enum):
    mpii_15 = "mpii-15"
    mpii_16 = "mpii-16"
    coco_17 = "coco-17"
    coco_18 = "coco-18"
    body_25 = "body-25"
    halpe_133 = "halpe-133"
    halpe_136 = "halpe-136"


class PoseModelConfigEnum(Enum):
    rtmpose_s_8xb256_420e_body8_256x192 = "rtmpose-s_8xb256-420e_body8-256x192"
    rtmpose_m_8xb256_420e_body8_256x192 = "rtmpose-m_8xb256-420e_body8-256x192"
    rtmpose_m_8xb256_420e_body8_384x288 = "rtmpose-m_8xb256-420e_body8-384x288"
    rtmpose_l_8xb256_420e_body8_256x192 = "rtmpose-l_8xb256-420e_body8-256x192"
    rtmpose_l_8xb256_420e_body8_384x288 = "rtmpose-l_8xb256-420e_body8-384x288"


class PoseModelCheckpointEnum(Enum):
    rtmpose_s_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_m_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_m_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_l_simcc_body7_pt_body7_420e_256x192_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )
    rtmpose_l_simcc_body7_pt_body7_420e_384x288_3f5a1437_20230504 = (
        "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504"
    )


class DetectorModelConfigEnum(Enum):
    rtmdet_nano_640_8xb32_coco_person = "rtmdet_nano_640-8xb32_coco-person"
    rtmdet_m_640_8xb32_coco_person = "rtmdet_m_640-8xb32_coco-person"


class DetectorModelCheckpointEnum(Enum):
    rtmdet_nano_8xb32_100e_coco_obj365_person_05d8511e = (
        "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e"
    )
    rtmdet_m_8xb32_100e_coco_obj365_person_235e8209 = (
        "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209"
    )


class Pose2dMetadataCommon(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    inference_run_id: UUID4
    environment_id: str
    classroom_date: date
    keypoints_format: KeypointsFormatEnum
    bounding_box_format: BoundingBoxFormatEnum
    pose_model_config: PoseModelConfigEnum
    pose_model_checkpoint: PoseModelCheckpointEnum
    detection_model_config: DetectorModelConfigEnum
    detection_model_checkpoint: DetectorModelCheckpointEnum

    @field_serializer("classroom_date")
    def serialize_classroom_date(self, dt: date, _info):
        return dt.strftime("%Y-%m-%d")


class Pose2dMetadata(Pose2dMetadataCommon):
    camera_device_id: str


def rounded_float(v: float) -> float:
    return round(v, 3)


RoundedFloat = Annotated[float, AfterValidator(rounded_float)]


class PoseOutput(BaseModel):
    keypoints: Tuple[
        Tuple[RoundedFloat, RoundedFloat, RoundedFloat, RoundedFloat], ...
    ]  # ((x, y, visibility, score), ...)


class BoundingBoxOutput(BaseModel):
    bbox: Tuple[
        RoundedFloat, RoundedFloat, RoundedFloat, RoundedFloat, RoundedFloat
    ]  # (x, y, x|w, y|h, score)


class Pose2d(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    timestamp: datetime
    metadata: Pose2dMetadata
    pose: PoseOutput
    bbox: BoundingBoxOutput
