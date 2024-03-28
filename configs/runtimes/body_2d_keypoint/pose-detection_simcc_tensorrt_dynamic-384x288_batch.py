_base_ = ['../mmdeploy/configs/mmpose/pose-detection_static.py', '../mmdeploy/configs/mmpose/../_base_/backends/tensorrt.py']

onnx_config = dict(
    input_shape=[288, 384],
    output_names=['simcc_x', 'simcc_y'],
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'simcc_x': {
            0: 'batch'
        },
        'simcc_y': {
            0: 'batch'
        }
    })

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 384, 288],
                    opt_shape=[100, 3, 384, 288],
                    max_shape=[100, 3, 384, 288])))
    ])
