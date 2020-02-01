CONFIG = {
    'DATA_PATH': '../../dataset/new_train_data',
    'SAVE_PATH': '../../result/temp/resnet.pt',
    'RECORD_PATH': '../../result/temp/resnet.rec',
    'PRETRAINED_PATH': None,
    'LEARNING_RATE': 0.0001,
    'INPUT_SIZE': 224,
    'BOTTLENECK_SIZE': 2048,
    'NUM_CLASS': 5,
    'BATCH_SIZE': 32,
    'EPOCHS': 100,
}

DATA_CONFIG = {
    'MEAN': (0.485, 0.456, 0.406),
    'STD': (0.229, 0.224, 0.225),
    'INITIAL_SAMPLING_WEIGHTS': [1] * CONFIG['NUM_CLASS'],
    'FINAL_SAMPLING_WEIGHTS': [1] * CONFIG['NUM_CLASS'],
    'DECAY_RATE': 1.0,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-30, 30),
        'translation_ratio': (20 / 224, 20 / 224),  # 20 pixel in the report
        'sigma': 0.5
    }
}
