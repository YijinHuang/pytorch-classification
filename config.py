import torchvision.models as models


BASIC_CONFIG = {
    'network': 'resnet50',  # shoud be one name in NET_CONFIG below
    'data_path': 'path/to/your/dataset',
    'data_index': None,  # alternative way to build dataset
    'save_path': 'path/to/save/folder',
    'record_path': 'path/to/save/log/folder',
    'pretrained': True,  # load pretrained parameters in ImageNet
    'checkpoint': None,  # load other pretrained model
    'num_classes': 5,  # number of categories
    'random_seed': 0,  # random seed for reproducibilty
    'device': 'cuda'  # 'cuda' / 'cpu'
}

DATA_CONFIG = {
    'input_size': 224,
    'mean': (0.485, 0.456, 0.406),  # for data normalization
    'std': (0.229, 0.224, 0.225),
    'sampling_strategy': 'shuffle',  # 'shuffle' / 'balance' / 'dynamic'
    'sampling_weights_decay_rate': 0.9,  # if sampling_strategy is dynamic, sampling weight will decay from balance to shuffle
}

TRAIN_CONFIG = {
    'epochs': 50,  # total training epochs
    'batch_size': 16,  # training batch size
    'optimizer': 'SGD',  # SGD / ADAM
    'criterion': 'CE',  # 'CE' / 'MSE', cross entropy or mean squared error. Generally, MSE is better than CE on kappa.
    'loss_weight': None,  # None / 'balance' / 'dynamic' / list with shape num_classes. Weights for loss function. Don't use it with weighted sampling!
    'loss_weight_decay_rate': 0.9,  # if loss_weights is dynamic, loss weight will decay from balance to equivalent weights
    'learning_rate': 0.001,  # initial learning rate
    'lr_scheduler': 'cosine',  # one str name in SCHEDULER_CONFIG below, scheduler configurations are in SCHEDULER_CONFIG.
    'momentum': 0.9,  # momentum for SGD optimizer
    'weight_decay': 0.0005,  # weight decay for SGD and ADAM
    'kappa_prior': True,  # save model with higher kappa or higher accuracy in validation set
    'warmup_epochs': 0,  # warmup epochs
    'num_workers': 16,  # number of cpus used to load data at each step
    'save_interval': 5,  # number of epochs to store model
    'pin_memory': True  # enables fast data transfer to CUDA-enabled GPUs
}

# you can add any networks in torchvision.models
NET_CONFIG = {
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'args': {}
}

# configuration for data augmentation
DATA_AUGMENTATION = {
    'brightness': 0.2,  # how much to jitter brightness
    'contrast': 0.2,  # How much to jitter contrast
    'scale': (1 / 1.15, 1.15),  # range of size of the origin size cropped
    'ratio': (0.7, 1.3),  # range of aspect ratio of the origin aspect ratio cropped
    'degrees': (-180, 180),  # range of degrees to select from
    'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
}

# you can add any learning rate scheduler in torch.optim.lr_scheduler
SCHEDULER_CONFIG = {
    'exponential': {
        'gamma': 0.6  # Multiplicative factor of learning rate decay
    },
    'multiple_steps': {
        'milestones': [15, 25, 45],  # List of epoch indices. Must be increasing
        'gamma': 0.1,  # Multiplicative factor of learning rate decay
    },
    'cosine': {
        'T_max': TRAIN_CONFIG['epochs'] - TRAIN_CONFIG['warmup_epochs'],  # Maximum number of iterations.
        'eta_min': 0  # Minimum learning rate.
    },
    'reduce_on_plateau': {
        'mode': 'min',  # In min mode, lr will be reduced when the quantity monitored has stopped decreasing
        'factor': 0.1,  # Factor by which the learning rate will be reduced
        'patience': 5,  # Number of epochs with no improvement after which learning rate will be reduced.
        'threshold': 1e-4,  # Threshold for measuring the new optimum
        'eps': 1e-5,  # Minimal decay applied to lr
    },
    'clipped_cosine': {
        'T_max': TRAIN_CONFIG['epochs'] - TRAIN_CONFIG['warmup_epochs'],
        'min_lr': 1e-4  # lr will stay as min_lr when achieve it
    }
}
