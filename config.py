import torchvision.models as models
from resnest import torch as resnest


BASE_CONFIG = {
    'NETWORK': 'resnext101_32x8d',  # shoud be name in NET_CONFIG
    'DATA_PATH': 'path/to/your/dataset',
    'DATA_INDEX': None,  # if not None, using this pickle file to build dataset
    'SAVE_PATH': 'path/to/save/folder',
    'RECORD_PATH': 'path/to/save/log/folder',
    'PRETRAINED': True,
    'CHECKPOINT': None,
    'NUM_CLASSES': 5,
    'RANDOM_SEED': 0
}

DATA_CONFIG = {
    'MEAN': (0.485, 0.456, 0.406),
    'STD': (0.229, 0.224, 0.225),
    'INITIAL_SAMPLING_WEIGHTS': [1] * BASE_CONFIG['NUM_CLASSES'],  # weighted sampling
    'FINAL_SAMPLING_WEIGHTS': [1] * BASE_CONFIG['NUM_CLASSES'],
    'DECAY_RATE': 1,  # if not 1, sampling weight will decay from initial to final
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7, 1.3),
        'ratation': (-180, 180),
        'translation_ratio': (0.2, 0.2)
    }
}

TRAIN_CONFIG = {
    'EPOCHS': 50,
    'BATCH_SIZE': 48,
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 0.0005,
    'KAPPA_PRIOR': True,  # save model with higher kappa or higher accuracy in validation set
    'WARMUP_EPOCH': 5,
    'NUM_WORKERS': 16,
    'SAVE_INTERVAL': 5,
    'NUM_CLASSES': BASE_CONFIG['NUM_CLASSES']
}

# you can add any networks in torchvision.models or customize in model.py
NET_CONFIG = {
    'resnet50': {
        'MODEL': models.resnet50,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.5,
        'OPTIONAL': {}
    },
    'resnet101': {
        'MODEL': models.resnet101,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.5,
        'OPTIONAL': {}
    },
    'inception_v3': {
        'MODEL': models.inception_v3,
        'INPUT_SIZE': 299,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.5,
        'OPTIONAL': {
            'aux_logits': False
        },
    },
    'resnext50_32x4d': {
        'MODEL': models.resnext50_32x4d,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.5,
        'OPTIONAL': {}
    },
    'resnext101_32x8d': {
        'MODEL': models.resnext101_32x8d,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.5,
        'OPTIONAL': {}
    },
    'resnest269': {
        'MODEL': resnest.resnest269,
        'INPUT_SIZE': 456,
        'BOTTLENECK_SIZE': 2048,
        'DROPOUT': 0.5,
        'OPTIONAL': {}
    },
    'efficientnet-b0': {'INPUT_SIZE': 224},
    'efficientnet-b1': {'INPUT_SIZE': 240},
    'efficientnet-b2': {'INPUT_SIZE': 260},
    'efficientnet-b3': {'INPUT_SIZE': 300},
    'efficientnet-b4': {'INPUT_SIZE': 380},
    'efficientnet-b5': {'INPUT_SIZE': 456},
}
