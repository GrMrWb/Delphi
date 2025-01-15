import numpy as np
import torch
from src.evaluation.utils import EvaluationMetrices, confusion_matrix
from torchmetrics.functional.classification import multiclass_calibration_error

def get_gold_labels(y, probabilities):
    y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=probabilities.shape[-1])
    
    # Now we extract the gold label and predicted true label probas

    # If the labels are binary [0,1]
    if len(torch.squeeze(y)) == 1:
        # get true labels
        true_probabilities = torch.tensor(probabilities)

        # get gold labels
        probabilities, y = torch.squeeze(
            torch.tensor(probabilities),
        ), torch.squeeze(y)

        batch_gold_label_probabilities = torch.where(
            y == 0,
            1 - probabilities,
            probabilities,
        )

    # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
    elif len(torch.squeeze(y)) == 2:
        # get true labels
        batch_true_probabilities = torch.max(probabilities)

        # get gold labels
        batch_gold_label_probabilities = torch.masked_select(
            probabilities,
            y.bool(),
        )
    else:

        # get true labels
        batch_true_probabilities = torch.max(probabilities)

        # get gold labels
        batch_gold_label_probabilities = torch.masked_select(
            probabilities,
            y.bool(),
        )

    # move torch tensors to cpu as np.arrays()
    gold_labels = (
        batch_gold_label_probabilities.detach().cpu().numpy()
    )
    batch_true_probabilities = batch_true_probabilities.detach().cpu().numpy()
    
    return gold_labels
    
def confidence(y, probabilities) -> np.ndarray:
    """
    Returns:
        Average predictive confidence across epochs: np.array(n_samples)
    """
    gold_labels = get_gold_labels(y, probabilities)
    
    return np.mean(gold_labels, axis=-1)

def aleatoric(y, probabilities):
    """
    Returns:
        Aleatric uncertainty of true label probability across epochs: np.array(n_samples): np.array(n_samples)
    """
    gold_labels = get_gold_labels(y, probabilities)
    
    preds = gold_labels
    return np.mean(preds * (1 - preds), axis=-1)

def variability(y, probabilities) -> np.ndarray:
    """
    Returns:
        Epistemic variability of true label probability across epochs: np.array(n_samples)
    """
    gold_labels = get_gold_labels(y, probabilities)
    
    return np.std(gold_labels, axis=-1)

def correctness(y, probabilities) -> np.ndarray:
    """
    Returns:
        Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
    """
    gold_labels = get_gold_labels(y, probabilities)
    
    return np.mean(gold_labels > 0.5, axis=-1)

def entropy(y, probabilities):
    """
    Returns:
        Predictive entropy of true label probability across epochs: np.array(n_samples)
    """
    gold_labels = get_gold_labels(y, probabilities)
    
    X = gold_labels
    return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

def entropy_without_gold_labels(X):
    return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

def mi(y, probabilities):
    preds = probabilities.cpu().detach().numpy()
    
    predictive_entropy = entropy_without_gold_labels(np.mean(preds, axis=0))
    
    expected_entropy = np.mean(entropy_without_gold_labels(preds),axis=0)
    
    return predictive_entropy - expected_entropy

def mi_BO(probabilities, target=None, y=None):
    preds = probabilities.cpu().detach().numpy()
    
    predictive_entropy = entropy_without_gold_labels(np.mean(preds, axis=0))
    
    # preds = target.cpu().detach().numpy()
    
    expected_entropy = np.mean(entropy_without_gold_labels(preds), axis=0)
    
    return predictive_entropy - expected_entropy

def evaluate_predictions(y, probabilities):
    evaluation = EvaluationMetrices().class_report(probabilities, y,True)
    
    results = {
        "accuracy"      : evaluation["accuracy"],
        "precision"     : evaluation['macro avg']["precision"],
        "recall"        : evaluation['macro avg']["recall"],
        "aleatoric"     : aleatoric(y, probabilities),
        "confidence"    : confidence(y, probabilities),
        "mi"            : mi(y, probabilities),
        "correctness"   : correctness(y, probabilities),
        "entropy"       : entropy(y, probabilities),
        "variability"   : variability(y, probabilities),
        "matrix"        : confusion_matrix(y.cpu().detach().numpy(), torch.max(probabilities.cpu().detach(), 1)[1].numpy()),
        "ece"           : multiclass_calibration_error(probabilities, y, num_classes=probabilities.shape[1], n_bins=5, norm='l1').item(),
        "mce"           : multiclass_calibration_error(probabilities, y, num_classes=probabilities.shape[1], n_bins=5, norm='max').item()
    }
    
    return results