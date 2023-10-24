python \
    "../scripts/pose_inference_demo.py" \
    "../configs/mmdet/rtmdet_m_640-8xb32_coco-person.py" \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input input/private_video/dahlia_example.mp4 \
    --output-root output/vis_results/ \
    --save-predictions
