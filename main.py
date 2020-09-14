import os
import random
import shutil

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import generate_model
from train import train, evaluate
from data_utils import generate_dataset
from config import BASE_CONFIG, NET_CONFIG


def main():
    # reproducibility
    seed = BASE_CONFIG['RANDOM_SEED']
    set_random_seed(seed)

    # create folder
    save_path = BASE_CONFIG['SAVE_PATH']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # create folder
    save_dir = BASE_CONFIG['SAVE_PATH']
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

    # create logger
    record_path = BASE_CONFIG['RECORD_PATH']
    if os.path.exists(record_path):
        shutil.rmtree(record_path) 
    logger = SummaryWriter(BASE_CONFIG['RECORD_PATH'])

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
