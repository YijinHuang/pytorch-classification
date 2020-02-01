import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_utils import ScheduledWeightedSampler
from metrics import accuracy, quadratic_weighted_kappa


def train(
    model,
    train_dataset,
    val_dataset,
    epochs,
    learning_rate,
    batch_size,
    save_path
):
    # create dataloader
    train_targets = [sampler[1] for sampler in train_dataset.imgs]
    weighted_sampler = ScheduledWeightedSampler(len(train_dataset), train_targets, True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=32, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, shuffle=False)

    # define loss and optimizier
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

    # learning rate warmup and decay
    warmup_epoch = 20
    warmup_batch = len(train_loader) * warmup_epoch
    remain_batch = len(train_loader) * (epochs - warmup_epoch)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remain_batch)
    warmup_scheduler = WarmupLRScheduler(optimizer, warmup_batch, learning_rate)

    # train
    record_epochs, accs, losses = _train(
        model,
        train_loader,
        val_loader,
        cross_entropy,
        optimizer,
        epochs,
        save_path,
        weighted_sampler=weighted_sampler,
        lr_scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
    )
    return model, record_epochs, accs, losses


def _train(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    epochs,
    save_path,
    weighted_sampler=None,
    lr_scheduler=None,
    warmup_scheduler=None
):
    model_dict = model.state_dict()
    trainable_layers = [(tensor, model_dict[tensor].size()) for tensor in model_dict]
    print_msg('Trainable layers: ', ['{}\t{}'.format(k, v) for k, v in trainable_layers])

    # angular loss
    angular_penalty = AngularPenalty(s=32)

    # train
    max_kappa = 0
    record_epochs, accs, losses = [], [], []
    model.train()
    for epoch in range(1, epochs + 1):
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # learning rate update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            if epoch > 1:
                curr_lr = optimizer.param_groups[0]['lr']
                print_msg('Learning rate warmup to {}'.format(curr_lr))
        elif lr_scheduler:
            if epoch % 10 == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print_msg('Current learning rate is {}'.format(curr_lr))

        total = 0
        correct = 0
        epoch_loss = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            if warmup_scheduler and not warmup_scheduler.is_finish():
                warmup_scheduler.step()
            elif lr_scheduler:
                lr_scheduler.step()

            X, y = train_data
            X, y = X.cuda(), y.long().cuda()

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

        # save model
        c_matrix = np.zeros((5, 5), dtype=int)
        acc = _eval(model, val_loader, c_matrix)
        kappa = quadratic_weighted_kappa(c_matrix)
        print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
        if kappa > max_kappa:
            torch.save(model, save_path)
            max_kappa = kappa
            print_msg('Model save at {}'.format(save_path))

        # record
        record_epochs.append(epoch)
        accs.append(acc)
        losses.append(avg_loss)

    return record_epochs, accs, losses


def evaluate(model_path, test_dataset):
    c_matrix = np.zeros((5, 5), dtype=int)

    trained_model = torch.load(model_path).cuda()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_acc = _eval(trained_model, test_loader, c_matrix)
    print('========================================')
    print('Finished! test acc: {}'.format(test_acc))
    print('Confusion Matrix:')
    print(c_matrix)
    print('quadratic kappa: {}'.format(quadratic_weighted_kappa(c_matrix)))
    print('========================================')


def _eval(model, dataloader, c_matrix=None):
    model.eval()
    torch.set_grad_enabled(False)

    correct = 0
    total = 0
    from tqdm import tqdm
    for step, test_data in enumerate(dataloader):
        X, y = test_data
        X, y = X.cuda(), y.long().cuda()

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


# for angular loss
class AngularPenalty(nn.Module):
    def __init__(self, s=64, m=0.5, eps=1e-7):
        super(AngularPenalty, self).__init__()

        self.s = s
        self.m = m
        self.eps = eps

    def forward(self, x, gt):
        thetas = torch.acos(torch.clamp(x, -1. + self.eps, 1 - self.eps))
        for i, y in enumerate(gt):
            thetas[i][y] += self.m
        logits = self.s * torch.cos(thetas)
        return logits


class WarmupLRScheduler:
    def __init__(self, optimizer, warmup_batch, initial_lr):
        self.step_num = 1
        self.optimizer = optimizer
        self.warmup_batch = warmup_batch
        self.initial_lr = initial_lr

    def step(self):
        if self.step_num <= self.warmup_batch:
            self.step_num += 1
            curr_lr = (self.step_num / self.warmup_batch) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def is_finish(self):
        return self.step_num > self.warmup_batch
