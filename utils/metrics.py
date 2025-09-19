import torch
import torcheval.metrics as tm
from utils.func import print_msg


class Estimator():
    def __init__(self, metrics, num_classes, criterion, thresholds=None):
        self.criterion = criterion
        self.num_classes = num_classes
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds

        if criterion in regression_based_metrics and 'auc' in metrics:
            metrics.remove('auc')
            print_msg('AUC is not supported for regression based metrics {}.'.format(criterion), warning=True)

        self.task = 'binary' if num_classes == 2 else 'multi_class'
        self.metrics = metrics
        self.metrics_fn = {}
        for m in metrics:
            metric, kargs = metrics_factory[m][self.task]
            if self.task == 'multi_class':
                kargs['num_classes'] = num_classes
            self.metrics_fn[m] = metric(**kargs)
        self.conf_mat_fn = tm.MulticlassConfusionMatrix(num_classes=num_classes)

    def update(self, predictions, targets):
        targets = targets.data.cpu().long()
        logits = predictions.data.cpu()
        predictions = self.to_prediction(logits)

        # update metrics
        self.conf_mat_fn.update(predictions, targets)
        for m in self.metrics_fn.keys():
            if m in logits_required_metrics:
                logits = logits if self.task == 'multi_class' else logits[:, 1]
                self.metrics_fn[m].update(logits, targets)
            else:
                self.metrics_fn[m].update(predictions, targets)

    def get_scores(self, digits=-1):
        scores = {m: self._compute(m, digits) for m in self.metrics}
        return scores

    def _compute(self, metric, digits=-1):
        score = self.metrics_fn[metric].compute().item()
        score = score if digits == -1 else round(score, digits)
        return score
    
    def get_conf_mat(self):
        return self.conf_mat_fn.compute().numpy().astype(int)

    def reset(self):
        for m in self.metrics_fn.keys():
            self.metrics_fn[m].reset()
        self.conf_mat_fn.reset()
    
    def to_prediction(self, predictions):
        if self.criterion in regression_based_metrics:
            predictions = torch.tensor([self.classify(p.item()) for p in predictions]).long()
        else:
            predictions = torch.argmax(predictions, dim=1).long()

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i


class QuadraticWeightedKappa():
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.conf_mat = torch.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, predictions, targets):
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1

    def compute(self):
        return self.quadratic_weighted_kappa(self.conf_mat)

    def reset(self):
        self.conf_mat = torch.zeros((self.num_classes, self.num_classes), dtype=int)

    def quadratic_weighted_kappa(self, conf_mat):
        assert conf_mat.shape[0] == conf_mat.shape[1]
        cate_num = conf_mat.shape[0]

        # Quadratic weighted matrix
        weighted_matrix = torch.zeros((cate_num, cate_num))
        for i in range(cate_num):
            for j in range(cate_num):
                weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

        # Expected matrix
        ground_truth_count = torch.sum(conf_mat, axis=1)
        pred_count = torch.sum(conf_mat, axis=0)
        expected_matrix = torch.outer(ground_truth_count, pred_count)

        # Normalization
        conf_mat = conf_mat / conf_mat.sum()
        expected_matrix = expected_matrix / expected_matrix.sum()

        observed = (conf_mat * weighted_matrix).sum()
        expected = (expected_matrix * weighted_matrix).sum()
        return (observed - expected) / (1 - expected)


metrics_factory = {
    'acc': {
        'binary': (tm.BinaryAccuracy, dict()),
        'multi_class': (tm.MulticlassAccuracy, dict())
    },
    'f1': {
        'binary': (tm.BinaryF1Score, dict()),
        'multi_class': (tm.MulticlassF1Score, dict(average='macro'))
    },
    'auc': {
        'binary': (tm.BinaryAUROC, dict()),
        'multi_class': (tm.MulticlassAUROC, dict())
    },
    'precision': {
        'binary': (tm.BinaryPrecision, dict()),
        'multi_class': (tm.MulticlassPrecision, dict(average='macro'))
    },
    'recall': {
        'binary': (tm.BinaryRecall, dict()),
        'multi_class': (tm.MulticlassRecall, dict(average='macro'))
    },
    'kappa': {
        'binary': (QuadraticWeightedKappa, dict()),
        'multi_class': (QuadraticWeightedKappa, dict())
    }
}
available_metrics = metrics_factory.keys()
logits_required_metrics = ['auc']
regression_based_metrics = ['mean_square_error', 'mean_absolute_error', 'smooth_L1']
