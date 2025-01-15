import torch
import torch.nn as nn
from kornia.losses.focal import FocalLoss

def choose_loss_fn(loss_fn='crossentropy', label_smoothing=0.0, alpha=1, gamma=0.75):
    """
    Choose a loss function:
        1. CrossEntropy or crossentropy
        2. BinaryCrossEntropy or bce
        3. MeanSquare or mse
        4. NLL or nll
        4. Focal Loss
    By default is crossentropy
    """
    if loss_fn == 'CrossEntropy' or loss_fn == 'crossentropy':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_fn == 'BinaryCrossEntropy' or loss_fn == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_fn == 'MeanSquare' or loss_fn == 'mse':
        return nn.MSELoss()
    elif loss_fn == 'NLL' or loss_fn == 'nll':
        return nn.NLLLoss()
    elif loss_fn == 'FL' or loss_fn == 'focal_loss':
        return FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

def choose_optimizer(parameters, optimizer='ADAM',learning_rate=0.001):
    """
    Choose an Optimizer:
        1. ADAM or adam
        2. SGD or sgd
        3. LBFGS or lbfgs
        4. custom or Custom
    By default is adam

    Choose the value of learning_rate which by default is set to 0.001 
    """
    if optimizer == 'SGD' or optimizer == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate) #,momentum=0.9)
    elif optimizer == 'ADAM' or optimizer == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer == 'LBFGS' or optimizer == 'lbfgs':
        return torch.optim.LBFGS(parameters, lr=learning_rate,weight_decay=0.1,momentum=0.1)
