import torch
import numpy as np
from config import BASIC_CONFIG


def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)


def accuracy(predictions, targets, c_matrix=None):
    predictions = predictions.data
    targets = targets.data

    # avoid modifying origin predictions
    predicted = torch.tensor(
        [torch.argmax(p) for p in predictions]
    ).to(BASIC_CONFIG['DEVICE']).long()

    # update confusion matrix
    if c_matrix is not None:
        for i, p in enumerate(predicted):
            c_matrix[int(targets[i])][int(p.item())] += 1

    correct = (predicted == targets).sum().item()
    return correct / len(predicted)
