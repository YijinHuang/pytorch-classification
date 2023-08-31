import os
import sys
import time
import random
import pickle
import argparse

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, '..')
from utils.func import *
from utils.metrics import Estimator
from modules.builder import build_torchvision_model
from fundus.fusion_modules import FusionModel, PairedEyeDataset


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet50', help='model architecture')
parser.add_argument('--data-index', type=str, default=None, help='paired image data index path')
parser.add_argument('--save-path', type=str, default='./checkpoints', help='save path')
parser.add_argument('--encoder', type=str, default=None, help='the model trained using single eye dataset')
parser.add_argument('--num-classes', type=int, default=5, help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=str, default='cuda', help='device')

parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--input-size', type=int, default=512, help='input size')
parser.add_argument('--learning-rate', type=float, default=0.02, help='learning rate')
parser.add_argument('--criterion', type=str, default='mean_square_error', help='mean_square_error / cross_entropy')
parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='number of workers')
parser.add_argument('--eval-interval', type=int, default=1, help='the epoch interval of evaluating model on val dataset')
parser.add_argument('--save-interval', type=int, default=5, help='the epoch interval of saving model')
parser.add_argument('--disable-progress', action='store_true', help='disable progress bar')


def main():
    args = parser.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    set_random_seed(args.seed)
    model = generate_model(args)
    train_dataset, test_dataset, val_dataset = generate_dataset(args)
    estimator = Estimator(['acc', 'kappa'], args.num_classes, args.criterion)
    train(
        args=args,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator
    )

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(save_path, 'best_validation_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    evaluate(args, model, checkpoint, test_dataset, estimator)


def train(args, model, train_dataset, val_dataset, estimator):
    device = args.device
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay
    )
    loss_function = nn.MSELoss() if args.criterion == 'mean_square_error' else nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # start training
    model.train()
    max_indicator = 0
    avg_loss = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader)) if not args.disable_progress else enumerate(train_loader)
        for step, train_data in progress:
            X_1, X_2, y = train_data
            X_1, X_2, y = X_1.to(device), X_2.to(device), y.to(device).float()

            y_pred = model(X_1, X_2)
            loss = loss_function(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)

            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            message = '[{}] epoch: [{} / {}], loss: {:.6f}'.format(current_time, epoch + 1, args.epochs, avg_loss)
            if not args.disable_progress:
                progress.set_description(message)

        if args.disable_progress:
            print(message)

        train_scores = estimator.get_scores(4)
        scores_txt = ', '.join(['{}: {}'.format(metric, score) for metric, score in train_scores.items()])
        print('Training metrics:', scores_txt)

        # validation performance
        if epoch % args.eval_interval == 0:
            eval(model, val_loader, estimator, device)
            val_scores = estimator.get_scores(6)
            scores_txt = ['{}: {}'.format(metric, score) for metric, score in val_scores.items()]
            print_msg('Validation metrics:', scores_txt)

            # save model
            indicator = val_scores['kappa']
            if indicator > max_indicator:
                torch.save(
                    model.state_dict(), 
                    os.path.join(args.save_path, 'best_validation_weights.pt')
                )
                max_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(args.save_path))

        if epoch % args.save_interval == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(args.save_path, 'epoch_{}.pt'.format(epoch))
            )

    # save final model
    torch.save(
        model.state_dict(), 
        os.path.join(args.save_path, 'final_weights.pt')
    )


def evaluate(args, model, checkpoint, test_dataset, estimator):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    print('Running on Test set...')
    eval(model, test_loader, estimator, args.device)

    print('================Finished================')
    test_scores = estimator.get_scores(6)
    for metric, score in test_scores.items():
        print('{}: {}'.format(metric, score))
    print('Confusion Matrix:')
    print(estimator.get_conf_mat())
    print('========================================')


def eval(model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X_1, X_2, y = test_data
        X_1, X_2, y = X_1.to(device), X_2.to(device), y.to(device).float()

        y_pred = model(X_1, X_2)
        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


def generate_model(args):
    arch = args.arch
    out_features = select_out_features(
        args.num_classes,
        args.criterion
    )
    encoder = build_torchvision_model(arch, out_features)
    weights = torch.load(args.encoder)
    encoder.load_state_dict(weights, strict=True)

    model = FusionModel(encoder)
    model.to(args.device)
    return model


def generate_dataset(args):
    train_transform, test_transform = data_transforms(args)

    data = pickle.load(open(args.data_index, 'rb'))
    train_set, test_set, val_set = data['train'], data['test'], data['val']

    train_dataset = PairedEyeDataset(train_set, train_transform)
    test_dataset = PairedEyeDataset(test_set, test_transform)
    val_dataset = PairedEyeDataset(val_set, test_transform)
    dataset = train_dataset, test_dataset, val_dataset

    print_dataset_info(dataset)
    return dataset


def data_transforms(args):
    mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
    std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(
            size=(args.input_size, args.input_size),
            scale=(0.87, 1.15),
            ratio=(0.7, 1.3)
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
    ]
    normalization = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        *normalization
    ])
    return train_preprocess, test_preprocess


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
