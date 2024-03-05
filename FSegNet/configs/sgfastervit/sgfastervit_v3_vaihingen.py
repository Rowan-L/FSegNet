_base_ = [
    '../_base_/models/sgfastervit.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(depths=[3, 3, 12, 5], dim=128, in_dim=64, resolution=[512, 512], drop_path_rate=0.3, window_size=[7, 7, 16, 8],
                  layer_scale=1e-5),
    decode_head=dict(in_channels=[128, 256, 512, 1024], channels=128, num_classes=6))


work_dir = "Model save directory"
