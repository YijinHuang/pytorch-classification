import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_utils import ScheduledWeightedSampler, LossWeightsScheduler
from metrics import accuracy, quadratic_weighted_kappa
from config import BASIC_CONFIG, TRAIN_CONFIG, DATA_CONFIG, SCHEDULER_CONFIG


def train(model, train_dataset, val_dataset, save_path, logger):
    # define weighted_sampler
    sampling_strategy = DATA_CONFIG['SAMPLING_STRATEGY']
    if sampling_strategy == 'BALANCE':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'DYNAMIC':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, DATA_CONFIG['SAMPLING_WEIGHTS_DECAY_RATE'])
    else:
        weighted_sampler = None

    # define data loader
    batch_size = TRAIN_CONFIG['BATCH_SIZE']
    num_workers = TRAIN_CONFIG['NUM_WORKERS']
    pin_memory = TRAIN_CONFIG['PIN_MEMORY']
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
    weight = None
    loss_weight_scheduler = None
    loss_weight = TRAIN_CONFIG['LOSS_WEIGHT']
    if loss_weight == 'BALANCE':
        loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
    elif loss_weight == 'DYNAMIC':
        loss_weight_scheduler = LossWeightsScheduler(train_dataset, TRAIN_CONFIG['LOSS_WEIGHT_DECAY_RATE'])
    elif isinstance(loss_weight, list):
        assert len(loss_weight) == len(train_dataset.classes)
        weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=BASIC_CONFIG['DEVICE'])
    cross_entropy = nn.CrossEntropyLoss(weight=weight)

    # define optmizer
    optimizer_strategy = TRAIN_CONFIG['OPTIMIZER']
    learning_rate = TRAIN_CONFIG['LEARNING_RATE']
    weight_decay = TRAIN_CONFIG['WEIGHT_DECAY']
    momentum = TRAIN_CONFIG['MOMENTUM']
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise Exception('Not implemented optimizer.')

    # define learning rate scheduler
    warmup_epochs = TRAIN_CONFIG['WARMUP_EPOCHS']
    scheduler_strategy = TRAIN_CONFIG['LR_SCHEDULER']
    scheduler_config = SCHEDULER_CONFIG[scheduler_strategy]
    if scheduler_strategy == 'COSINE':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'MULTIPLE_STEPS':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
    elif scheduler_strategy == 'REDUCE_ON_PLATEAU':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
    elif scheduler_strategy == 'REDUCE_ON_PLATEAU':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)
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
        cross_entropy,
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
    device = BASIC_CONFIG['DEVICE']
    epochs = TRAIN_CONFIG['EPOCHS']
    num_classes = TRAIN_CONFIG['NUM_CLASSES']
    kappa_prior = TRAIN_CONFIG['KAPPA_PRIOR']

    # print configuration
    print_msg('Basic configuration: ', ['{}:\t{}'.format(k, v) for k, v in BASIC_CONFIG.items()])
    print_msg('Data configuration: ', ['{}:\t{}'.format(k, v) for k, v in DATA_CONFIG.items()])
    print_msg('Training configuration: ', ['{}:\t{}'.format(k, v) for k, v in TRAIN_CONFIG.items()])

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

        curr_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0:
            print_msg('Current learning rate is {}'.format(curr_lr))

        total = 0
        correct = 0
        epoch_loss = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device).long()

            # forward
            y_pred = model(X)
            loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            total += y.size(0)
            correct += accuracy(y_pred, y) * y.size(0)
            avg_loss = epoch_loss / (step + 1)
            avg_acc = correct / total
            progress.set_description(
                'epoch: {}, loss: {:.6f}, acc: {:.4f}'
                .format(epoch, avg_loss, avg_acc)
            )

        # validation performance
        c_matrix = np.zeros((num_classes, num_classes), dtype=int)
        acc = _eval(model, val_loader, c_matrix)
        kappa = quadratic_weighted_kappa(c_matrix)
        print('validation accuracy: {}, kappa: {}'.format(acc, kappa))

        # save model
        indicator = kappa if kappa_prior else acc
        if indicator > max_indicator:
            torch.save(model, os.path.join(save_path, 'best_validation_model.pt'))
            max_indicator = indicator
            print_msg('Best in validation set. Model save at {}'.format(save_path))

        if epoch % TRAIN_CONFIG['SAVE_INTERVAL'] == 0:
            torch.save(model, os.path.join(save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if TRAIN_CONFIG['LR_SCHEDULER'] == 'REDUCE_ON_PLATEAU':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        logger.add_scalar('training loss', avg_loss, epoch)
        logger.add_scalar('training accuracy', avg_acc, epoch)
        logger.add_scalar('learning rate', curr_lr, epoch)
        logger.add_scalar('validation accuracy', acc, epoch)
        logger.add_scalar('validation kappa', kappa, epoch)

    # save final model
    torch.save(model, os.path.join(save_path, 'final_model.pt'))
    logger.close()


def evaluate(model_path, test_dataset):
    device = BASIC_CONFIG['DEVICE']
    num_classes = TRAIN_CONFIG['NUM_CLASSES']
    batch_size = TRAIN_CONFIG['BATCH_SIZE']
    num_workers = TRAIN_CONFIG['NUM_WORKERS']

    trained_model = torch.load(model_path).to(device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print('Running on Test set...')
    c_matrix = np.zeros((num_classes, num_classes), dtype=int)
    test_acc = _eval(trained_model, test_loader, c_matrix)

    print('========================================')
    print('Finished! test acc: {}'.format(test_acc))
    print('Confusion Matrix:')
    print(c_matrix)
    print('quadratic kappa: {}'.format(quadratic_weighted_kappa(c_matrix)))
    print('========================================')


def _eval(model, dataloader, c_matrix=None):
    device = BASIC_CONFIG['DEVICE']

    model.eval()
    torch.set_grad_enabled(False)

    total = 0
    correct = 0
    for test_data in dataloader:
        X, y = test_data
        X, y = X.to(device), y.long().to(device)

        y_pred = model(X)
        total += y.size(0)
        correct += accuracy(y_pred, y, c_matrix) * y.size(0)
    acc = round(correct / total, 4)

    model.train()
    torch.set_grad_enabled(True)
    return acc


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


# reference: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WarmupLRScheduler():
    def __init__(self, optimizer, warmup_epochs, initial_lr, verbose=True):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.verbose = verbose

    def step(self):
        if self.epoch <= self.warmup_epochs:
            self.epoch += 1
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr
        if self.verbose:
            print_msg('Warming up. Current learning rate is {}'.format(curr_lr))

    def is_finish(self):
        return self.epoch >= self.warmup_epochs
