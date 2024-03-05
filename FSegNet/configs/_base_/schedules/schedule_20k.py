# optimizer
# optimizer = dict(type='SGD', lr=0.00005, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
# potsdam 864       vaihingen 86
iters = 86*2
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=iters*3),
    dict(
        type='PolyLR',
        power=1.0,
        begin=iters*3,
        end=iters*100,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iters*100, val_interval=iters)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=iters, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
