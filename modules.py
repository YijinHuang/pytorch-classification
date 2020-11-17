import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler


class CustomizedModel(nn.Module):
    def __init__(self, name, backbone, num_classes, pretrained=False, **kwargs):
        super(CustomizedModel, self).__init__()

        net = backbone(pretrained=pretrained, **kwargs)
        if 'resnet' in name or 'resnext' in name or 'shufflenet' in name:
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif 'densenet' in name:
            net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif 'vgg' in name:
            net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        elif 'mobilenet' in name:
            net.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(net.last_channel, num_classes),
            )
        elif 'squeezenet' in name:
            net.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            raise NotImplementedError('Not implemented network.')
        self.net = net

    def forward(self, x):
        x = self.net(x)
        x = x.squeeze()
        return x


# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
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
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr

    def step(self):
        if self.epoch <= self.warmup_epochs:
            self.epoch += 1
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def is_finish(self):
        return self.epoch >= self.warmup_epochs


class ScheduledWeightedSampler(Sampler):
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset
        self.decay_rate = decay_rate

        self.num_samples = len(dataset)
        self.targets = [sample[1] for sample in dataset.imgs]
        self.class_weights = self.cal_class_weights(self.targets)

        self.epoch = 0
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.double)
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.double)
        self.sample_weight = torch.zeros(self.num_samples, dtype=torch.double)
        for i, _class in enumerate(self.targets):
            self.sample_weight[i] = self.w0[_class]

    def step(self):
        if self.decay_rate < 1:
            self.epoch += 1
            factor = self.decay_rate**(self.epoch - 1)
            self.weights = factor * self.w0 + (1 - factor) * self.wf
            for i, _class in enumerate(self.targets):
                self.sample_weight[i] = self.weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.sample_weight, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

    def cal_class_weights(self, train_targets):
        num_classes = len(self.dataset.classes)
        classes_idx = list(range(num_classes))
        class_count = [self.targets.count(i) for i in classes_idx]
        weights = [self.num_samples / class_count[i] for i in classes_idx]
        min_weight = min(weights)
        class_weights = [weights[i] / min_weight for i in classes_idx]
        return class_weights


class LossWeightsScheduler():
    def __init__(self, dataset, decay_rate):
        self.dataset = dataset
        self.decay_rate = decay_rate

        self.num_samples = len(dataset)
        self.targets = [sample[1] for sample in dataset.imgs]
        self.class_weights = self.cal_class_weights(self.targets)

        self.epoch = 0
        self.w0 = torch.as_tensor(self.class_weights, dtype=torch.float32)
        self.wf = torch.as_tensor([1] * len(self.dataset.classes), dtype=torch.float32)

    def step(self):
        weights = self.w0
        if self.decay_rate < 1:
            self.epoch += 1
            factor = self.decay_rate**(self.epoch - 1)
            weights = factor * self.w0 + (1 - factor) * self.wf
        return weights

    def __len__(self):
        return self.num_samples

    def cal_class_weights(self, train_targets):
        num_classes = len(self.dataset.classes)
        classes_idx = list(range(num_classes))
        class_count = [self.targets.count(i) for i in classes_idx]
        weights = [self.num_samples / class_count[i] for i in classes_idx]
        min_weight = min(weights)
        class_weights = [weights[i] / min_weight for i in classes_idx]
        return class_weights


class ClippedCosineAnnealingLR():
    def __init__(self, optimizer, T_max, min_lr):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        self.min_lr = min_lr
        self.finish = False

    def step(self):
        if not self.finish:
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            if curr_lr < self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.min_lr
                self.finish = True

    def is_finish(self):
        return self.finish


# http://defauw.ai/diabetic-retinopathy-detection/
class KappaLoss():
    def __init__(self, num_classes, y_pow=2, eps=1e-10):
        self.num_classes = num_classes
        self.y_pow = y_pow
        self.eps = eps

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).cuda()
        y_true = y[y_true]

        y_true = y_true.float()
        repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).cuda()
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / (self.eps + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(torch.reshape(hist_rater_a, [num_classes, 1]), torch.reshape(hist_rater_b, [1, num_classes]))
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.eps)
