import os
import sys
import random
import shutil

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint

from config import *
from metrics import Estimator
from train import train, evaluate
from utils import print_msg, generate_dataset, generate_model


def main():
    # print configuration
    print_config()

    # reproducibility
    seed = BASIC_CONFIG['random_seed']
    set_random_seed(seed)

    # create folder
    save_path = BASIC_CONFIG['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # build model
    network = BASIC_CONFIG['network']
    device = BASIC_CONFIG['device']
    criterion = TRAIN_CONFIG['criterion']
    num_classes = BASIC_CONFIG['num_classes']
    out_features = 1 if criterion == 'MSE' else num_classes
    model = generate_model(
        network,
        out_features,
        NET_CONFIG,
        device,
        BASIC_CONFIG['pretrained'],
        BASIC_CONFIG['checkpoint']
    )

    # create dataset
    train_dataset, test_dataset, val_dataset = generate_dataset(
        DATA_CONFIG,
        BASIC_CONFIG['data_path'],
        BASIC_CONFIG['data_index']
    )

    # create logger
    record_path = BASIC_CONFIG['record_path']
    if os.path.exists(record_path):
        shutil.rmtree(record_path)
    logger = SummaryWriter(BASIC_CONFIG['record_path'])

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
    model_path = os.path.join(save_path, 'best_validation_model.pt')
    evaluate(model_path, TRAIN_CONFIG, test_dataset, num_classes, estimator, device)
    print('This is the performance of the final model:')
    model_path = os.path.join(save_path, 'final_model.pt')
    evaluate(model_path, TRAIN_CONFIG, test_dataset, num_classes, estimator, device)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_config():
    print('=========================')
    print('===Basic configuration===')
    pprint(BASIC_CONFIG, sort_dicts=False)
    print('===Data configuration===')
    pprint(DATA_CONFIG, sort_dicts=False)
    print('===Training configuration===')
    pprint(TRAIN_CONFIG, sort_dicts=False)
    print('========================')


if __name__ == '__main__':
    main()
