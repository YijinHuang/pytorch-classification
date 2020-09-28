import torchvision.models as models


BASIC_CONFIG = {
    'NETWORK': 'resnet50',  # shoud be one name in NET_CONFIG
    'DATA_PATH': 'path/to/your/dataset',
    'DATA_INDEX': None,  # alternative way to build dataset
    'SAVE_PATH': 'path/to/save/folder',
    'RECORD_PATH': 'path/to/save/log/folder',
    'PRETRAINED': True,  # load pretrained parameters in ImageNet
    'CHECKPOINT': None,  # load other pretrained model
    'NUM_CLASSES': 5,  # number of categories
    'RANDOM_SEED': 0,  # random seed for reproducibilty
    'DEVICE': 'cuda'  # 'cuda' / 'cpu'
}

DATA_CONFIG = {
    'MEAN': (0.485, 0.456, 0.406),  # for data normalization
    'STD': (0.229, 0.224, 0.225),
    'SAMPLING_STRATEGY': 'SHUFFLE',  # 'SHUFFLE' / 'BALANCE' / 'DYNAMIC'
    'SAMPLING_WEIGHTS_DECAY_RATE': 0.9,  # if SAMPLING_STRATEGY is DYNAMIC, sampling weight will decay from balance to shuffle
}

TRAIN_CONFIG = {
    'EPOCHS': 50,  # total training epochs
    'BATCH_SIZE': 16,  # training batch size
    'OPTIMIZER': 'SGD',  # SGD / ADAM
    'LOSS_WEIGHT': None,  # None / 'BALANCE' / 'DYNAMIC' / list with shape NUM_CLASSES. Weights for loss function. Don't use it with weighted sampling!
    'LOSS_WEIGHT_DECAY_RATE': 0.9,  # if LOSS_WEIGHTS is DYNAMIC, loss weight will decay from balance to equivalent weights
    'LEARNING_RATE': 0.001,  # initial learning rate
    'LR_SCHEDULER': 'COSINE',  # 'EXPONENTIAL' / 'MULTIPLE_STEPS' / 'COSINE' / 'REDUCE_ON_PLATEAU', scheduler configurations are in SCHEDULER_CONFIG.
    'MOMENTUM': 0.9,  # momentum for SGD optimizer
    'WEIGHT_DECAY': 0.0005,  # weight decay for SGD and ADAM
    'KAPPA_PRIOR': True,  # save model with higher kappa or higher accuracy in validation set
    'WARMUP_EPOCHS': 0,  # warmup epochs
    'NUM_WORKERS': 16,  # number of cpus used to load data at each step
    'SAVE_INTERVAL': 5,  # number of epochs to store model
    'PIN_MEMORY': True,  # enables fast data transfer to CUDA-enabled GPUs
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
    'EXPONENTIAL': {
        'gamma': 0.6  # Multiplicative factor of learning rate decay
    },
    'MULTIPLE_STEPS': {
        'milestones': [15, 25, 45],  # List of epoch indices. Must be increasing
        'gamma': 0.1,  # Multiplicative factor of learning rate decay
    },
    'COSINE': {
        'T_max': TRAIN_CONFIG['EPOCHS'] - TRAIN_CONFIG['WARMUP_EPOCHS'],  # Maximum number of iterations.
        'eta_min': 0  # Minimum learning rate.
    },
    'REDUCE_ON_PLATEAU': {
        'mode': 'min',  # In min mode, lr will be reduced when the quantity monitored has stopped decreasing
        'factor': 0.1,  # Factor by which the learning rate will be reduced
        'patience': 5,  # Number of epochs with no improvement after which learning rate will be reduced.
        'threshold': 1e-4,  # Threshold for measuring the new optimum
        'eps': 1e-5,  # Minimal decay applied to lr
    }
}
