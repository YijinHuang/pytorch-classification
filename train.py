import os
import sys

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributed import all_reduce, all_gather, ReduceOp

from utils.func import *
from modules.loss import *
from modules.scheduler import *


def train(cfg, model, train_dataset, val_dataset, estimator, logger=None):
    device = cfg.base.device
    optimizer = initialize_optimizer(cfg, model)
    train_sampler, val_sampler = initialize_sampler(cfg, train_dataset, val_dataset)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(cfg, optimizer)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset, train_sampler, val_sampler)

    # start training
    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        # resampling weight update
        if cfg.dist.distributed:
            train_sampler.set_epoch(epoch)
        elif train_sampler:
            train_sampler.step()

        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader)) if cfg.base.progress else enumerate(train_loader)
        for step, train_data in progress:
            X, y = train_data
            X = X.cuda(cfg.dist.gpu) if cfg.dist.distributed else X.to(device)
            y = y.cuda(cfg.dist.gpu) if cfg.dist.distributed else y.to(device)
            y = select_target_type(y, cfg.train.criterion)

            # forward
            y_pred = model(X)
            loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            if cfg.dist.distributed:
                all_reduce(loss, ReduceOp.SUM)
                loss = loss / cfg.dist.world_size

                y_pred_list = [torch.zeros_like(y_pred) for _ in range(cfg.dist.world_size)]
                y_list = [torch.zeros_like(y) for _ in range(cfg.dist.world_size)]
                all_gather(y_pred_list, y_pred)
                all_gather(y_list, y)
                y_pred = torch.cat(y_pred_list, dim=0)
                y = torch.cat(y_list, dim=0)

            if is_main(cfg):
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (step + 1)
                estimator.update(y_pred, y)
                avg_acc = estimator.get_accuracy(6)
                avg_kappa = estimator.get_kappa(6)


                message = 'epoch: [{} / {}], loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}'\
                        .format(epoch, cfg.train.epochs, avg_loss, avg_acc, avg_kappa)
                if cfg.base.progress:
                    progress.set_description(message)
            
        if is_main(cfg) and not cfg.base.progress:
            print(message)

        if is_main(cfg) and cfg.train.sample_view:
            samples = torchvision.utils.make_grid(X)
            samples = inverse_normalize(samples, cfg.data.mean, cfg.data.std)
            logger.add_image('input samples', samples, epoch, dataformats='CHW')

        # validation performance
        if epoch % cfg.train.eval_interval == 0:
            eval(cfg, model, val_loader, cfg.train.criterion, estimator, device)
            acc = estimator.get_accuracy(6)
            kappa = estimator.get_kappa(6)
            print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
            if is_main(cfg) and logger:
                logger.add_scalar('validation accuracy', acc, epoch)
                logger.add_scalar('validation kappa', kappa, epoch)

            # save model
            indicator = kappa if cfg.train.kappa_prior else acc
            if is_main(cfg) and indicator > max_indicator:
                save_weights(model, os.path.join(cfg.base.save_path, 'best_validation_weights.pt'))
                max_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(cfg.base.save_path))

        if is_main(cfg) and epoch % cfg.train.save_interval == 0:
            save_weights(model, os.path.join(cfg.base.save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if cfg.solver.lr_scheduler == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        if is_main(cfg) and logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('training accuracy', avg_acc, epoch)
            logger.add_scalar('training kappa', avg_kappa, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    if is_main(cfg):
        save_weights(model, os.path.join(cfg.base.save_path, 'final_weights.pt'))

    if is_main(cfg) and logger:
        logger.close()


def evaluate(cfg, model, test_dataset, estimator):
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if cfg.dist.distributed else None
    test_loader = DataLoader(
        test_dataset,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory
    )

    print('Running on Test set...')
    eval(cfg, model, test_loader, cfg.train.criterion, estimator, cfg.base.device)

    if is_main(cfg):
        print('========================================')
        print('Finished! test acc: {}'.format(estimator.get_accuracy(6)))
        print('Confusion Matrix:')
        print(estimator.conf_mat)
        print('quadratic kappa: {}'.format(estimator.get_kappa(6)))
        print('========================================')


def eval(cfg, model, dataloader, criterion, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X = X.cuda(cfg.dist.gpu) if cfg.dist.distributed else X.to(device)
        y = y.cuda(cfg.dist.gpu) if cfg.dist.distributed else y.to(device)
        y = select_target_type(y, criterion)

        y_pred = model(X)

        if cfg.dist.distributed:
            y_pred_list = [torch.zeros_like(y_pred) for _ in range(cfg.dist.world_size)]
            y_list = [torch.zeros_like(y) for _ in range(cfg.dist.world_size)]
            all_gather(y_pred_list, y_pred)
            all_gather(y_list, y)
            y_pred = torch.cat(y_pred_list, dim=0)
            y = torch.cat(y_list, dim=0)
        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


# define weighted_sampler
def initialize_sampler(cfg, train_dataset, val_dataset):
    sampler = None
    sampling_strategy = cfg.data.sampling_strategy

    if cfg.dist.distributed:
        if sampling_strategy != 'instance_balanced':
            msg = 'Resampling is not allowed when distributed parallel is applied. \
                   Please set sampling_strategy to instance_balanced.'
            exit_with_error(msg)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=cfg.dist.world_size,
            rank=cfg.dist.rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=cfg.dist.world_size,
            rank=cfg.dist.rank
        )
    else:
        val_sampler = None
        if sampling_strategy == 'class_balanced':
            train_sampler = ScheduledWeightedSampler(train_dataset, 1)
        elif sampling_strategy == 'progressively_balanced':
            train_sampler = ScheduledWeightedSampler(train_dataset, cfg.data.sampling_weights_decay_rate)
        elif sampling_strategy == 'instance_balanced':
            train_sampler = None
        else:
            raise NotImplementedError('Not implemented resampling strategy.')

    return train_sampler, val_sampler


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset, train_sampler, val_sampler):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# define loss and loss weights scheduler
def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion
    criterion_args = cfg.criterion_args[criterion]

    weight = None
    loss_weight_scheduler = None
    loss_weight = cfg.train.loss_weight
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.train.loss_weight_decay_rate)
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
        loss = nn.CrossEntropyLoss(weight=weight, **criterion_args)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss(**criterion_args)
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss(**criterion_args)
    elif criterion == 'smooth_L1':
        loss = nn.SmoothL1Loss(**criterion_args)
    elif criterion == 'kappa_loss':
        loss = KappaLoss(**criterion_args)
    elif criterion == 'focal_loss':
        loss = FocalLoss(**criterion_args)
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg.solver.optimizer
    learning_rate = cfg.solver.learning_rate
    weight_decay = cfg.solver.weight_decay
    momentum = cfg.solver.momentum
    nesterov = cfg.solver.nesterov
    adamw_betas = cfg.solver.adamw_betas

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
    elif optimizer_strategy == 'ADAMW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=adamw_betas,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(cfg, optimizer):
    warmup_epochs = cfg.train.warmup_epochs
    learning_rate = cfg.solver.learning_rate
    scheduler_strategy = cfg.solver.lr_scheduler

    if not scheduler_strategy:
        lr_scheduler = None
    else:
        scheduler_args = cfg.scheduler_args[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_strategy == 'clipped_cosine':
            lr_scheduler = ClippedCosineAnnealingLR(optimizer, **scheduler_args)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
