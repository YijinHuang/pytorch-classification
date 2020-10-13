import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from modules import *
from metrics import *


def train(model, train_dataset, val_dataset, save_path, logger):
    # define weighted_sampler
    sampling_strategy = DATA_CONFIG['sampling_strategy']
    if sampling_strategy == 'balance':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'dynamic':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, DATA_CONFIG['sampling_weights_decay_rate'])
    else:
        weighted_sampler = None

    # define data loader
    batch_size = TRAIN_CONFIG['batch_size']
    num_workers = TRAIN_CONFIG['num_workers']
    pin_memory = TRAIN_CONFIG['pin_memory']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False if weighted_sampler else True,
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    # define loss and loss weights scheduler
    criterion = TRAIN_CONFIG['criterion']
    weight = None
    loss_weight_scheduler = None
    loss_weight = TRAIN_CONFIG['loss_weight']
    if criterion == 'CE':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, TRAIN_CONFIG['loss_weight_decay_rate'])
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=BASIC_CONFIG['device'])
        loss_function = nn.CrossEntropyLoss(weight=weight)
    elif criterion == 'MSE':
        loss_function = nn.MSELoss()
    else:
        raise NotImplementedError('Not implemented loss function.')

    # define optmizer
    optimizer_strategy = TRAIN_CONFIG['optimizer']
    learning_rate = TRAIN_CONFIG['learning_rate']
    weight_decay = TRAIN_CONFIG['weight_decay']
    momentum = TRAIN_CONFIG['momentum']
    nesterov = TRAIN_CONFIG['nesterov']
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    # define learning rate scheduler
    warmup_epochs = TRAIN_CONFIG['warmup_epochs']
    scheduler_strategy = TRAIN_CONFIG['lr_scheduler']
    if scheduler_strategy is not None:
        scheduler_config = SCHEDULER_CONFIG[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
        elif scheduler_strategy == 'clipped_cosine':
            lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_config)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')
    else:
        lr_scheduler = None

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    # train
    _train(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        save_path,
        logger,
        weighted_sampler,
        lr_scheduler,
        warmup_scheduler,
        loss_weight_scheduler
    )


def _train(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    save_path,
    logger=None,
    weighted_sampler=None,
    lr_scheduler=None,
    warmup_scheduler=None,
    loss_weight_scheduler=None
):
    device = BASIC_CONFIG['device']
    epochs = TRAIN_CONFIG['epochs']
    criterion = TRAIN_CONFIG['criterion']
    num_classes = BASIC_CONFIG['num_classes']
    kappa_prior = TRAIN_CONFIG['kappa_prior']

    # print configuration
    print_msg('Basic configuration: ', ['{}:\t{}'.format(k, v) for k, v in BASIC_CONFIG.items()])
    print_msg('Data configuration: ', ['{}:\t{}'.format(k, v) for k, v in DATA_CONFIG.items()])
    print_msg('Training configuration: ', ['{}:\t{}'.format(k, v) for k, v in TRAIN_CONFIG.items()])

    # intitial estimator
    estimator = Estimator(criterion, num_classes)

    # training
    max_indicator = 0
    model.train()
    for epoch in range(1, epochs + 1):
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step().to(device)
            loss_function.weight = weight

        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device)
            y = y.long() if criterion == 'CE' else y.float()

            # forward
            y_pred = model(X)
            loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)
            avg_acc = estimator.get_accuracy(6)
            avg_kappa = estimator.get_kappa(6)

            progress.set_description(
                'epoch: {}, loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}'
                .format(epoch, avg_loss, avg_acc, avg_kappa)
            )

        # validation performance
        eval(model, val_loader, estimator)
        acc = estimator.get_accuracy(6)
        kappa = estimator.get_kappa(6)
        print('validation accuracy: {}, kappa: {}'.format(acc, kappa))

        # save model
        indicator = kappa if kappa_prior else acc
        if indicator > max_indicator:
            torch.save(model, os.path.join(save_path, 'best_validation_model.pt'))
            max_indicator = indicator
            print_msg('Best in validation set. Model save at {}'.format(save_path))

        if epoch % TRAIN_CONFIG['save_interval'] == 0:
            torch.save(model, os.path.join(save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if TRAIN_CONFIG['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        logger.add_scalar('training loss', avg_loss, epoch)
        logger.add_scalar('training accuracy', avg_acc, epoch)
        logger.add_scalar('training kappa', avg_kappa, epoch)
        logger.add_scalar('learning rate', curr_lr, epoch)
        logger.add_scalar('validation accuracy', acc, epoch)
        logger.add_scalar('validation kappa', kappa, epoch)

    # save final model
    torch.save(model, os.path.join(save_path, 'final_model.pt'))
    logger.close()


def evaluate(model_path, test_dataset):
    device = BASIC_CONFIG['device']
    criterion = TRAIN_CONFIG['criterion']
    num_classes = BASIC_CONFIG['num_classes']
    batch_size = TRAIN_CONFIG['batch_size']
    num_workers = TRAIN_CONFIG['num_workers']

    trained_model = torch.load(model_path).to(device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print('Running on Test set...')
    estimator = Estimator(criterion, num_classes)
    eval(trained_model, test_loader, estimator)

    print('========================================')
    print('Finished! test acc: {}'.format(estimator.get_accuracy(6)))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    print('quadratic kappa: {}'.format(estimator.get_kappa(6)))
    print('========================================')


def eval(model, dataloader, estimator):
    device = BASIC_CONFIG['device']
    criterion = TRAIN_CONFIG['criterion']

    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X, y = X.to(device), y.to(device)
        y = y.long() if criterion == 'CE' else y.float()

        y_pred = model(X)
        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)
