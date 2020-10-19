import os
import sys
import random
import shutil

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import *
from train import train, evaluate
from modules import generate_model
from data_utils import generate_dataset


def main():
    # reproducibility
    seed = BASIC_CONFIG['random_seed']
    set_random_seed(seed)

    # create folder
    save_path = BASIC_CONFIG['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # build model
    network = BASIC_CONFIG['network']
    model = generate_model(
        network,
        BASIC_CONFIG['num_classes'],
        BASIC_CONFIG['checkpoint'],
        BASIC_CONFIG['pretrained']
    )

    # load dataset
    train_dataset, test_dataset, val_dataset = generate_dataset(
        BASIC_CONFIG['data_path'],
        BASIC_CONFIG['data_index']
    )

    # create logger
    record_path = BASIC_CONFIG['record_path']
    if os.path.exists(record_path):
        shutil.rmtree(record_path)
    logger = SummaryWriter(BASIC_CONFIG['record_path'])

    # train
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=save_path,
        logger=logger
    )

    # test
    print('This is the performance of the best validation model:')
    evaluate(os.path.join(save_path, 'best_validation_model.pt'), test_dataset)
    print('This is the performance of the final model:')
    evaluate(os.path.join(save_path, 'final_model.pt'), test_dataset)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
