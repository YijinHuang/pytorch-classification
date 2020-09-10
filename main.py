import os
import random

import torch
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import generate_model
from train import train, evaluate
from data_utils import generate_dataset
from config import BASE_CONFIG, DATA_CONFIG, NET_CONFIG


def main():
    save_dir = os.path.split(BASE_CONFIG['SAVE_PATH'])[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # build model
    network = BASE_CONFIG['NETWORK']
    model = generate_model(
        network,
        BASE_CONFIG['NUM_CLASSES'],
        BASE_CONFIG['CHECKPOINT'],
        BASE_CONFIG['PRETRAINED']
    )

    # load dataset
    train_dataset, test_dataset, val_dataset = generate_dataset(
        BASE_CONFIG['DATA_PATH'],
        NET_CONFIG[network]['INPUT_SIZE'],
        BASE_CONFIG['DATA_INDEX']
    )

    # train
    save_path = BASE_CONFIG['SAVE_PATH']
    logger = SummaryWriter(BASE_CONFIG['RECORD_PATH'])
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=save_path,
        logger=logger
    )

    # test
    print('This is the performance of best validation model:')
    evaluate(os.path.join(save_path, 'best_validation_model.pt'), test_dataset)
    print('This is the performance of final model:')
    evaluate(os.path.join(save_path, 'final_model.pt'), test_dataset)


if __name__ == '__main__':
    random.seed(1)
    torch.set_seed(1)
    main()
