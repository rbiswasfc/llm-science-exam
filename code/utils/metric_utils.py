# References:
# 1. https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking/notebook
# 2. https://www.kaggle.com/code/takamichitoda/llm-perplexity-ranking-ensemble

import numpy as np
import torch
import torch.nn as nn


class Perplexity(nn.Module):
    """
    Reference: https://www.youtube.com/watch?v=NURcDHhYe98
    """

    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)

        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity.cpu().item()


def precision_at_k(r, k):
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k


def get_score(predictions, truths):
    """
    Score is mean average precision at 3
    """

    n_questions = len(predictions)
    score = 0.0

    for u in range(n_questions):
        user_preds = predictions[u]
        user_true = truths[u]

        user_results = [1 if item == user_true else 0 for item in user_preds]

        for k in range(min(len(user_preds), 3)):
            score += precision_at_k(user_results, k+1) * user_results[k]
    score /= n_questions
    score = round(score, 4)

    return score

# ------


def _compute_metrics(true_ids, pred_ids, debug=False):
    """
    fbeta score for one example
    """

    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    # calculate the confusion matrix variables
    tp = len(true_ids.intersection(pred_ids))
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    # metrics
    f1 = tp / (tp + 0.5 * fp + 0.5*fn)
    f2 = tp / (tp + 0.2 * fp + 0.8*fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if debug:
        print("Ground truth count:", len(true_ids))
        print("Predicted count:", len(pred_ids))
        print("True positives:", tp)
        print("False positives:", fp)
        print("F2:", f2)

    to_return = {
        "f1": f1,
        "f2": f2,
        "precision": precision,
        "recall": recall,
    }

    return to_return


def average_fbeta_score(true_ids, pred_ids):
    """
    fbeta metric for learning equality - content recommendation task

    :param true_ids: ground truth content ids
    :type true_ids: List[List[str]]
    :param pred_ids: prediction content ids
    :type pred_ids: List[List[str]]
    """
    assert len(true_ids) == len(pred_ids), "length mismatch between truths and predictions"
    n_examples = len(true_ids)
    f2_scores = []

    for i in range(n_examples):
        ex_true_ids = true_ids[i]
        ex_pred_ids = pred_ids[i]
        if len(ex_pred_ids) == 0:
            f2 = 0
        else:
            f2 = _compute_metrics(ex_true_ids, ex_pred_ids)["f2"]
        f2_scores.append(f2)
    f2_score = np.mean(f2_scores)

    return f2_score


def compute_retrieval_metrics(true_ids, pred_ids):
    """
    fbeta metric for learning equality - content recommendation task

    :param true_ids: ground truth content ids
    :type true_ids: List[List[str]]
    :param pred_ids: prediction content ids
    :type pred_ids: List[List[str]]
    """
    assert len(true_ids) == len(pred_ids), "length mismatch between truths and predictions"
    n_examples = len(true_ids)
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []

    for i in range(n_examples):
        ex_true_ids = true_ids[i]
        ex_pred_ids = pred_ids[i]
        if len(ex_pred_ids) == 0:
            f1 = 0.
            f2 = 0.
            precision = 0.
            recall = 0.
        else:
            m = _compute_metrics(ex_true_ids, ex_pred_ids)
            f1 = m["f1"]
            f2 = m["f2"]
            precision = m["precision"]
            recall = m["recall"]

        f1_scores.append(f1)
        f2_scores.append(f2)
        precision_scores.append(precision)
        recall_scores.append(recall)

    to_return = {
        "f1_score": np.mean(f1_scores),
        "f2_score": np.mean(f2_scores),
        "precision_score": np.mean(precision_scores),
        "recall_score": np.mean(recall_scores),
    }

    return to_return
