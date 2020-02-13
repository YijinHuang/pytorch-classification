import torchvision.models as models


BASE_CONFIG = {
    'NETWORK': 'inception_v3',  # shoud be name in NET_CONFIG
    'DATA_PATH': 'path/to/your/dataset',
    'SAVE_PATH': 'path/to/save/model',
    'RECORD_PATH': 'path/to/save/log',
    'PRETRAINED': True,
    'CHECKPOINT': None,
    'NUM_CLASS': 5,
}

DATA_CONFIG = {
    'MEAN': (0.485, 0.456, 0.406),
    'STD': (0.229, 0.224, 0.225),
    'INITIAL_SAMPLING_WEIGHTS': [1] * BASE_CONFIG['NUM_CLASS'],  # make data sampling balance
    'FINAL_SAMPLING_WEIGHTS': [1] * BASE_CONFIG['NUM_CLASS'],
    'DECAY_RATE': 1.0,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7, 1.3),
        'ratation': (-30, 30),
        'translation_ratio': (0.1, 0.1),
        'sigma': 0.5  # ooptional, for color augmentation
    }
}

TRAIN_CONFIG = {
    'EPOCHS': 10,
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 0.0001,
    'WEIGHT_DECAY': 0.0005,
    'KAPPA_PRIOR': True,  # save model with higher kappa or higher accuracy
    'WARMUP_EPOCH': 20,
    'NUM_WORKERS': 32,
    'NUM_CLASS': BASE_CONFIG['NUM_CLASS']
}

# you can add any networks in torchvision.models
NET_CONFIG = {
    'resnet50': {
        'MODEL': models.resnet50,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.2,
        'OPTIONAL': {}
    },
    'resnet101': {
        'MODEL': models.resnet101,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.2,
        'OPTIONAL': {}
    },
    'inception_v3': {
        'MODEL': models.inception_v3,
        'INPUT_SIZE': 299,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.2,
        'OPTIONAL': {
            'aux_logits': False
        },
    },
    'efficientnet-b0': {'INPUT_SIZE': 224},
    'efficientnet-b1': {'INPUT_SIZE': 240},
    'efficientnet-b2': {'INPUT_SIZE': 260},
    'efficientnet-b3': {'INPUT_SIZE': 300},
    'efficientnet-b4': {'INPUT_SIZE': 380},
    'efficientnet-b5': {'INPUT_SIZE': 456},
}
