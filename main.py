import os

import pickle
import torch
import numpy as np

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
    model, record_epochs, accs, losses = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=BASE_CONFIG['SAVE_PATH']
    )
    pickle.dump(
        (record_epochs, accs, losses),
        open(BASE_CONFIG['RECORD_PATH'], 'wb')
    )

    # test
    evaluate(BASE_CONFIG['SAVE_PATH'], test_dataset)


if __name__ == '__main__':
    main()
