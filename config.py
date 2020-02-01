CONFIG = {
    'DATA_PATH': '../../dataset/train_data_full_512',
    'SAVE_PATH': '../../result/kaggle/resnet.pt',
    'RECORD_PATH': '../../result/kaggle/resnet.rec',
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
    'INITIAL_SAMPLING_WEIGHTS': [1.35742396, 14.2898356, 6.74819802, 42.5767116, 46.1766938],
    'FINAL_SAMPLING_WEIGHTS': [1, 2, 2, 2, 2],
    'DECAY_RATE': 0.95,
    'DATA_AUGMENTATION': {
        'scale': (1 / 1.15, 1.15),
        'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
        'ratation': (-30, 30),
        'translation_ratio': (20 / 224, 20 / 224),  # 20 pixel in the report
        'sigma': 0.5
    }
}