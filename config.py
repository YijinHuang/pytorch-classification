import torchvision.models as models
from utils import auto_statistics


BASIC_CONFIG = {
    'network': 'shufflenet_1_0',  # shoud be one name in NET_CONFIG below
    'data_path': '../../dataset/split_eyepacs_hq_512',
    'data_index': None,  # alternative way to build dataset
    'save_path': '../../result/eyepacs_small/temp',
    'record_path': '../../result/eyepacs_small/auto/temp',
    'pretrained': True,  # load pretrained parameters in ImageNet
    'checkpoint': None,  # load other pretrained model
    'num_classes': 5,  # number of categories
    'random_seed': 0,  # random seed for reproducibilty
    'device': 'cuda'  # 'cuda' / 'cpu'
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
    'nesterov': True,  # nesterov for SGD optimizer
    'weight_decay': 0.0005,  # weight decay for SGD and ADAM
    'kappa_prior': True,  # save model with higher kappa or higher accuracy in validation set
    'warmup_epochs': 0,  # warmup epochs
    'num_workers': 16,  # number of cpus used to load data at each step
    'save_interval': 5,  # number of epochs to store model
    'pin_memory': True  # enables fast data transfer to CUDA-enabled GPUs
}

# remove it if you want to use your own mean and std for normalization
mean, std = auto_statistics(
    BASIC_CONFIG['data_path'],
    BASIC_CONFIG['data_index'],
    TRAIN_CONFIG['batch_size'],
    TRAIN_CONFIG['num_workers']
)
DATA_CONFIG = {
    'input_size': 224,
    'mean': mean,  # list with three numbers for RGB
    'std': std,  # list with three numbers for RGB
    'sampling_strategy': 'shuffle',  # 'shuffle' / 'balance' / 'dynamic'
    'sampling_weights_decay_rate': 0.9,  # if sampling_strategy is dynamic, sampling weight will decay from balance to shuffle
    'data_augmentation': {
        'brightness': 0.2,  # how much to jitter brightness
        'contrast': 0.2,  # How much to jitter contrast
        'scale': (1 / 1.15, 1.15),  # range of size of the origin size cropped
        'ratio': (0.7, 1.3),  # range of aspect ratio of the origin aspect ratio cropped
        'degrees': (-180, 180),  # range of degrees to select from
        'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
    }
}

# you can add any networks in torchvision.models
NET_CONFIG = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
    'args': {}
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
