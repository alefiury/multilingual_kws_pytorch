import torch
from sklearn import metrics


def model_accuracy(label, output):
    """
    Calculates the model accuracy considering its outputs.

    ----
    Params:
        output: Model outputs.
        label: True labels.

    Returns:
        Accuracy as a decimal.
    """
    pb = output
    infered = torch.argmax(pb, dim=-1)
    equals = label == infered

    return torch.mean(equals.type(torch.FloatTensor))


def model_f1_score(label, output):
    """
    Calculates the model f1-score considering its outputs.

    ----
    Params:
        output: Model outputs.
        label: True labels.

    Returns:
        F1 score as a decimal.
    """
    pb = output
    infered = torch.argmax(pb, dim=-1)

    return metrics.f1_score(label.detach().cpu().numpy(), infered.detach().cpu().numpy(), average='macro')