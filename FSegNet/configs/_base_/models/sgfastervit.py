# model settings
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='FasterViT',
        ct_size=2,
        depths=[1, 3, 8, 5],
        dim=80,
        drop_path_rate=0.2,
        hat=[False, False, True, False],
        in_dim=32,
        mlp_ratio=4,
        num_heads=[2, 4, 8, 16],
        resolution=[512, 512],
        layer_scale=None,
        window_size=[7, 7, 7, 7]),
    decode_head=dict(
        type='LightHead',
        in_channels=[80, 160, 320, 640],
        in_index=[0, 1, 2, 3],
        channels=80,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

