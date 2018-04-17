def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


ensemble_models = {
    'stage1_test': {
        'rcnn': [
        ],
        'unet': [
            "/data/public/rw/kaggle-data-science-bowl/submissions/lb525_ensemble_s80px(0.5-2.0)xflip/submission_lb525.pkl"
        ]
    },
    'stage1_unet': {
        'rcnn': [
            "/data/public/rw/datasets/dsb2018/pickles/pickled_mask_info_full.pkl"
        ],
        'unet': [
            "/data/public/rw/kaggle-data-science-bowl/submissions/lb525_ensemble_s80px(0.5-2.0)xflip/submission_lb525.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold1_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold2_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold3_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold4_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold5_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold6_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold7_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
        ]
    },

    'stage2_unetv1': {
        'rcnn': [
            "/data/public/rw/datasets/dsb2018/pickles/pickled_mask_info_full_stage2.pkl"
        ],
        'unet': [
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_best496_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold1_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold2_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold3_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold4_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold5_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold6_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold7_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
        ]
    },
    'stage2_unetv1_norcnn': {
        'rcnn': [
        ],
        'unet': [
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_best496_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold1_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold2_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold3_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold4_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold5_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold6_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv1_fold7_ext2thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
        ]
    },

    'stage2_unetv2': {
        'rcnn': [
            "/data/public/rw/datasets/dsb2018/pickles/pickled_mask_info_full_stage2.pkl"
        ],
        'unet': [
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_best539_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold1_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold2_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold3_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold4_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold5_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold6_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold7_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
        ]
    },
    'stage2_unetv2_norcnn': {
        'rcnn': [
        ],
        'unet': [
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_best539_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold1_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold2_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold3_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold4_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold5_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold6_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
            "/data/public/rw/kaggle-data-science-bowl/submissions/stage2_w/unetv2_fold7_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
        ]
    },
}