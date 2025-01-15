import copy
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

# from src.delphi.utils import confidence_level

class PTFedProxLoss(_Loss):
    def __init__(self, mu: float = 0.01) -> None:
        """Compute FedProx loss: a loss penalizing the deviation from global model.
        Args:
            mu: weighting parameter
        """
        super().__init__()
        if mu < 0.0:
            raise ValueError("mu should be no less than 0.0")
        self.mu = mu

    def forward(self, input, target) -> torch.Tensor:
        """Forward pass in training.
        Args:
            input (nn.Module): the local pytorch model
            target (nn.Module): the copy of global pytorch model when local clients received it
                                at the beginning of each local round
        Returns:
            FedProx loss term
        """
        prox_loss: torch.Tensor = 0.0
        for param, ref in zip(input.named_parameters(), target.named_parameters()):
            prox_loss += (self.mu / 2) * torch.sum((param[1] - ref[1]) ** 2)

        return prox_loss
class pFedMeOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = copy.deepcopy(local_weight_updated)
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p, localweight = p.to('cuda'), localweight.to('cuda')
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']
    
class PerturbedGradientDescent(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])

class CustomCrossEntropy(nn.Module):
    r"""Criterion that computes our custom Cross Entropy.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        reduction: Specifies the reduction to apply to the 
            output: ``'none'`` | ``'mean'`` | ``'sum'``
          
        no smoothing is required

    Shape:
        - Input: : logits
        - Target: : the index
    """
    def __init__(self,
        alpha:float = 1,
        reduction: str = 'mean',
        target_confidence:float = 0.75
    ):
        super().__init__()
        self.alpha: float = alpha
        self.reduction: str = reduction
        self.target_confidence = target_confidence
    
    def forward(self, output: torch.Tensor, target: torch.Tensor)-> torch.Tensor:
        return custom_cross_entropy(output, target, self.alpha, self.reduction, self.target_confidence)
    
def custom_cross_entropy(
    output: torch.Tensor, 
    target: torch.Tensor, 
    alpha: float = 1,
    reduction: str = 'mean',
    target_confidence: float = 0.75
) -> torch.Tensor:
    
    assert output.device == target.device, f"input and target must be in the same device. Got: {output.device} and {target.device}"
    assert output.shape[0] == target.shape[0], f"Output's shape is different than target's shape. Make sure the target class is one hot"
    
    log_output_soft: torch.Tensor = output.log_softmax(1)
    
    one_hot_target = confidence_level(target, target_confidence=target_confidence, classes=output.shape[1])
    one_hot_target = one_hot_target.to(target.device)
    
    initial_loss = -alpha*log_output_soft
        
    loss_tmp = torch.einsum('bc...,bc...->b...', (one_hot_target, initial_loss))
    
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    
    return loss

def confidence_level(y: torch.Tensor, target_confidence : float =0.75, classes = 10 ):
    """Create an array with the desired output for all the actual outputs
    
    Args:
        y (Tensor): the input class
        target_confidence (float, optional): desired confidence for a class. Defaults to 0.75.

    Returns:
        Tensor: The desired output
    """
    try:
        target = torch.zeros(y.shape[0], classes)
    except:
        target = torch.zeros(1, classes)
        
    rest_of_class = (1 - target_confidence)/classes
    
    if len(target) > 1:
        for sample, real in zip(target, y):
            for batch, a in enumerate(sample):
                if batch != real.cpu().detach().numpy():
                    sample[batch] = rest_of_class 
                else:
                    sample[batch] = target_confidence            
    else:
        for i in range(10):
            if i != y.cpu().detach().numpy():
                target[0][i] = rest_of_class 
            else:
                target[0][i] = target_confidence
                
    return target
    