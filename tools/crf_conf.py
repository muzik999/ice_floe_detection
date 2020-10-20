default_conf = {
    'filter_size': 11,
    'blur': 4,
    'merge': True,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 1,
    "weight_init": 0.2,

    'trainable': False,
    'convcomp': False,
    'logsoftmax': True,  # use logsoftmax for numerical stability
    'softmax': True,
    'final_softmax': False,

    'pos_feats': {
        'sdims': 3,
        'compat': 3,
    },
    'col_feats': {
        'sdims': 80,
        'schan': 13,   # schan depend on the input scale.
                       # use schan = 13 for images in [0, 255]
                       # for normalized images in [-0.5, 0.5] try schan = 0.1
        'compat': 10,
        'use_bias': False
    },
    "trainable_bias": False,

    "pyinn": False
}