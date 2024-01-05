backend_config = dict(
    type='onnxruntime',
    precision='fp16',
    common_config=dict(
        min_positive_val=1e-7,
        max_finite_val=1e4,
        keep_io_types=False,
        disable_shape_infer=False,
        op_block_list=None,
        node_block_list=None))
