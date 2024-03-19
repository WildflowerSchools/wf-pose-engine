_base_ = ['../_base_/base_dynamic.py', '../mmdeploy/configs/_base_/backends/tensorrt.py']

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[100, 3, 640, 640],
                    opt_shape=[100, 3, 640, 640],
                    max_shape=[100, 3, 640, 640])))
    ])
