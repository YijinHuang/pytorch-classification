import os

import pickle
import torch
import numpy as np
import torchvision.models as models

from config import CONFIG
from model import *
from train import train, evaluate
from data_utils import generate_data

torch.set_num_threads(16)


def main():
    # load dataset
    train_dataset, test_dataset, val_dataset = generate_data(
        CONFIG['DATA_PATH'],
        CONFIG['INPUT_SIZE']
    )

    save_dir = os.path.split(CONFIG['SAVE_PATH'])[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pretrained_path = CONFIG['PRETRAINED_PATH']
    if pretrained_path:
        model = torch.load(CONFIG['PRETRAINED_PATH'])
        print('Load weights form {}'.format(pretrained_path))
    else:
        model = MyModel(models.resnet, CONFIG['BOTTLENECK_SIZE'], CONFIG['NUM_CLASS'], True).cuda()
        # model = MyModel(models.inception_v3, 2048, 5, True, aux_logits=False).cuda()
        # model = MyEfficientNet('5', CONFIG['BOTTLENECK_SIZE'], CONFIG['NUM_CLASS'], pretrained=True).cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # train
    model, record_epochs, accs, losses = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=CONFIG['EPOCHS'],
        learning_rate=CONFIG['LEARNING_RATE'],
        batch_size=CONFIG['BATCH_SIZE'],
        save_path=CONFIG['SAVE_PATH']
    )
    pickle.dump(
        (record_epochs, accs, losses),
        open(CONFIG['RECORD_PATH'], 'wb')
    )

    # test
    evaluate(CONFIG['SAVE_PATH'], test_dataset)


if __name__ == '__main__':
    main()
