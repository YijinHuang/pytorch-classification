import os
import sys
import random
import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import *
from metrics import Estimator
from train import train, evaluate
from data import generate_dataset
from modules import generate_model
from utils import print_config, select_out_features


def main():
    # create folder
    save_path = BASIC_CONFIG['save_path']
    if os.path.exists(save_path):
        overwirte = input('Save path {} exists.\nDo you want to overwrite it? (y/n)\n'.format(save_path))
        if overwirte != 'y':
            sys.exit(0)
    else:
        os.makedirs(save_path)

    # create logger
    record_path = BASIC_CONFIG['record_path']
    record_path = os.path.join(record_path, 'log-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(record_path)

    # print configuration
    print_config({
        'BASIC CONFIG': BASIC_CONFIG,
        'DATA CONFIG': DATA_CONFIG,
        'TRAIN CONFIG': TRAIN_CONFIG
    })

    # reproducibility
    seed = BASIC_CONFIG['random_seed']
    set_random_seed(seed)

    # build model
    net_name = BASIC_CONFIG['network']
    backbone = NET_CONFIG[net_name]
    device = BASIC_CONFIG['device']
    criterion = TRAIN_CONFIG['criterion']
    num_classes = BASIC_CONFIG['num_classes']
    out_features = select_out_features(num_classes, criterion)
    model = generate_model(
        net_name,
        backbone,
        out_features,
        device,
        BASIC_CONFIG['pretrained'],
        BASIC_CONFIG['checkpoint'],
        NET_CONFIG['args']
    )

    # create dataset
    train_dataset, test_dataset, val_dataset = generate_dataset(
        DATA_CONFIG,
        BASIC_CONFIG['data_path'],
        BASIC_CONFIG['data_index'],
        TRAIN_CONFIG['batch_size'],
        TRAIN_CONFIG['num_workers']
    )

    # create estimator and then train
    estimator = Estimator(criterion, num_classes, device)
    train(
        model=model,
        train_config=TRAIN_CONFIG,
        data_config=DATA_CONFIG,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=save_path,
        estimator=estimator,
        device=device,
        logger=logger
    )

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(save_path, 'best_validation_weights.pt')
    evaluate(model, checkpoint, TRAIN_CONFIG, test_dataset, estimator, device)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    evaluate(model, checkpoint, TRAIN_CONFIG, test_dataset, estimator, device)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
