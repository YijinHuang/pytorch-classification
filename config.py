import torchvision.models as models


BASIC_CONFIG = {
    'NETWORK': 'resnext101_32x8d',  # shoud be one name in NET_CONFIG
    'DATA_PATH': '../../dataset/train_data_full_512',
    'DATA_INDEX': None,  # alternative way to build dataset
    'SAVE_PATH': '../../result/eyepacs/test',
    'RECORD_PATH': '../../result/eyepacs/log/test',
    'PRETRAINED': True,  # load pretrained parameters in ImageNet
    'CHECKPOINT': None,  # load other pretrained model
    'NUM_CLASSES': 5,  # number of categories
    'RANDOM_SEED': 1  # random seed for reproducibilty
}

DATA_CONFIG = {
    'MEAN': (0.485, 0.456, 0.406),  # for data normalization
    'STD': (0.229, 0.224, 0.225),
    'SAMPLING_STRATEGY': 'DYNAMIC',  # SHUFFLE / BALANCE / DYNAMIC
    'DECAY_RATE': 1,  # if SAMPLING_STRATEGY is DYNAMIC, sampling weight will decay from balance to shuffle
}

TRAIN_CONFIG = {
    'EPOCHS': 50,  # total training epochs
    'BATCH_SIZE': 48,  # training batch size
    'OPTIMIZER': 'SGD',  # SGD / ADAM
    'LEARNING_RATE': 0.001,  # initial learning rate
    'LR_SCHEDULER': 'MULTIPLE_STEPS',  # MULTIPLE_STEPS / COSINE / REDUCE_ON_PLATEAU, scheduler configurations are in SCHEDULER_CONFIG.
    'WEIGHT_DECAY': 0.0005,
    'KAPPA_PRIOR': True,  # save model with higher kappa or higher accuracy in validation set
    'WARMUP_EPOCHS': 5,  # warmup epochs
    'NUM_WORKERS': 16,  # number of cpus used to load data at each step
    'SAVE_INTERVAL': 5,  # number of epochs to store model
    'NUM_CLASSES': BASIC_CONFIG['NUM_CLASSES']
}

# you can add any networks in torchvision.models
NET_CONFIG = {
    'resnet50': {
        'MODEL': models.resnet50,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'OPTIONAL': {}
    },
    'resnet101': {
        'MODEL': models.resnet101,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'OPTIONAL': {}
    },
    'inception_v3': {
        'MODEL': models.inception_v3,
        'INPUT_SIZE': 299,
        'BOTTLENECK_SIZE': 2048,
        'OPTIONAL': {
            'aux_logits': False
        },
    },
    'resnext50_32x4d': {
        'MODEL': models.resnext50_32x4d,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'OPTIONAL': {}
    },
    'resnext101_32x8d': {
        'MODEL': models.resnext101_32x8d,
        'INPUT_SIZE': 224,
        'BOTTLENECK_SIZE': 2048,
        'OPTIONAL': {}
    }
}

# configuration for data augmentation
DATA_AUGMENTATION = {
    'scale': (1 / 1.15, 1.15),
    'stretch_ratio': (0.7, 1.3),
    'ratation': (-180, 180),
    'translation_ratio': (0.2, 0.2)
}

# you can add any learning rate scheduler in torch.optim.lr_scheduler
SCHEDULER_CONFIG = {
    'MULTIPLE_STEPS': {
        'milestones': [15, 25, 45],
        'gamma': 0.1,
    },
    'REDUCE_ON_PLATEAU': {
        'mode': 'min',
        'factor': 0.1,
        'patience': 5,
        'threshold': 1e-4,
        'eps': 1e-5,
    }
}
