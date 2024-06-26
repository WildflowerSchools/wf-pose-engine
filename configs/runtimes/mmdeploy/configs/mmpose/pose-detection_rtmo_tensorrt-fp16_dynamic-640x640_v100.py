_base_ = ['./pose-detection_static.py', '../_base_/backends/tensorrt-fp16.py']

onnx_config = dict(
    output_names=['dets', 'keypoints'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'dets': {
            0: 'batch',
        },
        'keypoints': {
            0: 'batch'
        }
    })

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 35),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    # 320 is the max batch size before running into Tensor size limits: "The volume of a tensor cannot exceed 2^31-1"
                    min_shape=[320, 3, 640, 640],
                    opt_shape=[320, 3, 640, 640],
                    max_shape=[320, 3, 640, 640])))
    ])

codebase_config = dict(
    post_processing=dict(
        score_threshold=0.05,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=2000,
        keep_top_k=50,
        background_label_id=-1,
    ))
