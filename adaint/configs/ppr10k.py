exp_name = 'ailut_ppr10k'

custom_imports=dict(
    imports=['adaint'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='AiLUT',
    n_ranks=5,
    n_vertices=33,
    en_adaint=True,
    en_adaint_share=False,
    backbone='res18', # 'tpami'
    pretrained=True,
    n_colors=3,
    sparse_factor=0.0001,
    smooth_factor=0,
    monotonicity_factor=10.0,
    recons_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(n_fix_iters=3329*5)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'PPR10K'
val_dataset_type = 'PPR10K'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='cv2',
        flag='unchanged'),
    dict(type='FlipChannels', keys=['lq']), # BGR->RGB
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb'),
    dict(type='RandomRatioCrop', keys=['lq', 'gt'], crop_ratio=(0.6, 1.0)),
    dict(type='Resize', keys=['lq', 'gt'], scale=(448, 448), backend='cv2', interpolation='bilinear'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='cv2',
        flag='unchanged'),
    dict(type='FlipChannels', keys=['lq']), # BGR->RGB
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path'])
]

target = 'a' # change this line (a/b/c) to use other groundtruths

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type=train_dataset_type,
        dir_lq='data/PPR10K/source_aug_6',
        dir_gt=f'data/PPR10K/target_{target}',
        ann_file='data/PPR10K/train_aug.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}.tif',
        filetmpl_gt='{}.tif'),
    # val
    val=dict(
        type=val_dataset_type,
        dir_lq='data/PPR10K/source',
        dir_gt=f'data/PPR10K/target_{target}',
        ann_file='data/PPR10K/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.tif',
        filetmpl_gt='{}.tif'),
    # test
    test=dict(
        type=val_dataset_type,
        dir_lq='data/PPR10K/source',
        dir_gt=f'data/PPR10K/target_{target}',
        ann_file='data/PPR10K/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}.tif',
        filetmpl_gt='{}.tif'),
)

# optimizer
optimizers = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0,
    betas=(0.9, 0.999),
    eps=1e-8,
    paramwise_cfg=dict(custom_keys={'adaint': dict(lr_mult=0.1)}))
lr_config = None

# learning policy
total_iters = 3329*200

checkpoint_config = dict(interval=3329, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=3329, save_image=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
