import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules import *
from utils import print_msg


def train(model, train_config, data_config, train_dataset, val_dataset, save_path, estimator, device, logger=None):
    criterion = train_config['criterion']
    optimizer = initialize_optimizer(train_config, model)
    weighted_sampler = initialize_sampler(data_config, train_dataset)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(train_config, optimizer)
    loss_function, loss_weight_scheduler = initialize_loss(train_config, train_dataset, device)
    train_loader, val_loader = initialize_dataloader(train_config, train_dataset, val_dataset, weighted_sampler)

    # start training
    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(1, train_config['epochs'] + 1):
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
        eval(model, val_loader, criterion, estimator, device)
        acc = estimator.get_accuracy(6)
        kappa = estimator.get_kappa(6)
        print('validation accuracy: {}, kappa: {}'.format(acc, kappa))

        # save model
        indicator = kappa if train_config['kappa_prior'] else acc
        if indicator > max_indicator:
            torch.save(model, os.path.join(save_path, 'best_validation_model.pt'))
            max_indicator = indicator
            print_msg('Best in validation set. Model save at {}'.format(save_path))

        if epoch % train_config['save_interval'] == 0:
            torch.save(model, os.path.join(save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if train_config['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('training accuracy', avg_acc, epoch)
            logger.add_scalar('training kappa', avg_kappa, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)
            logger.add_scalar('validation accuracy', acc, epoch)
            logger.add_scalar('validation kappa', kappa, epoch)

    # save final model
    torch.save(model, os.path.join(save_path, 'final_model.pt'))
    if logger:
        logger.close()


def evaluate(model_path, train_config, test_dataset, num_classes, estimator, device):
    trained_model = torch.load(model_path).to(device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        shuffle=False
    )

    print('Running on Test set...')
    eval(trained_model, test_loader, train_config['criterion'], estimator, device)

    print('========================================')
    print('Finished! test acc: {}'.format(estimator.get_accuracy(6)))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    print('quadratic kappa: {}'.format(estimator.get_kappa(6)))
    print('========================================')


def eval(model, dataloader, criterion, estimator, device):
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


# define weighted_sampler
def initialize_sampler(data_config, train_dataset):
    sampling_strategy = data_config['sampling_strategy']
    if sampling_strategy == 'balance':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'dynamic':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, data_config['sampling_weights_decay_rate'])
    else:
        weighted_sampler = None
    return weighted_sampler


# define data loader
def initialize_dataloader(train_config, train_dataset, val_dataset, weighted_sampler):
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    pin_memory = train_config['pin_memory']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
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

    return train_loader, val_loader


# define loss and loss weights scheduler
def initialize_loss(train_config, train_dataset, device):
    criterion = train_config['criterion']
    weight = None
    loss_weight_scheduler = None
    loss_weight = train_config['loss_weight']
    if criterion == 'CE':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, train_config['loss_weight_decay_rate'])
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=device)
        loss_function = nn.CrossEntropyLoss(weight=weight)
    elif criterion == 'MSE':
        loss_function = nn.MSELoss()
    else:
        raise NotImplementedError('Not implemented loss function.')

    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(train_config, model):
    optimizer_strategy = train_config['optimizer']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    nesterov = train_config['nesterov']
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

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(train_config, optimizer):
    learning_rate = train_config['learning_rate']
    warmup_epochs = train_config['warmup_epochs']
    scheduler_strategy = train_config['lr_scheduler']
    scheduler_config = train_config['scheduler_config']

    lr_scheduler = None
    if scheduler_strategy in scheduler_config.keys():
        scheduler_config = scheduler_config[scheduler_strategy]
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

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
