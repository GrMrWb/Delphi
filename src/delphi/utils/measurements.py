import torch
import torch.nn.functional as F
import numpy as np

from scipy.spatial.distance import jensenshannon

from art.metrics.metrics import wasserstein_distance as art_wd

from src.evaluation.uncertainty_quant import mi, mi_BO
from src.delphi.utils.weight_io import read_weights

def confidence_level(y: torch.Tensor, target_confidence : float =0.75, classes = 10 ):
    """Create an array with the desired output for all the actual outputs
    
    Args:
        y (Tensor): the input class
        target_confidence (float, optional): desired confidence for a class. Defaults to 0.75.

    Returns:
        Tensor: The desired output
    """

    
    rest_of_class = (1 - target_confidence)/classes
    
    if target_confidence==0:
        target_confidence = 0.145
        rest_of_class = (1 - target_confidence)/classes
    
    if len(y.shape) > 1:
        target = torch.zeros(y.shape[0], classes)
        if len(target) > 1:
            for sample, real in zip(target, y):
                real_idx = real.argmax().cpu().detach().numpy()
                for batch, a in enumerate(sample):
                    if batch != real_idx:
                        sample[batch] = rest_of_class 
                    else:
                        sample[batch] = target_confidence            
        else:
            target = torch.zeros(1, classes)
            real_idx = y.cpu().detach().argmax().item()
            for i in range(10):
                if i != real_idx:
                    target[0][i] = rest_of_class 
                else:
                    target[0][i] = target_confidence
    else:
        if len(y) > 1:
            target = torch.zeros(len(y), classes)
            for sample, real in zip(target, y):
                for batch, a in enumerate(sample):
                    check = real.cpu().detach().numpy() if not isinstance(real, np.integer) else real
                    if batch != check:
                        sample[batch] = rest_of_class 
                    else:
                        sample[batch] = target_confidence            
        else:
            target = torch.zeros(1, classes)
            real_idx = y.cpu().detach().argmax().item()
            for i in range(10):
                if i != real_idx:
                    target[0][i] = rest_of_class 
                else:
                    target[0][i] = target_confidence
                
    return target

def wasserstein_matrix(samples: torch.Tensor,target: torch.Tensor) -> torch.Tensor: 
    # sum_of = []
    
    # for p,q in zip(samples, target):
    #     wd = wasserstein_distance(p,q)
    #     sum_of.append(wd)
    #     print(wd)
    # print(wd)
    
    sum_of = []
    for p,q in zip(samples, target):
        p, q = p.cpu().detach().numpy(), q.cpu().detach().numpy()
        wd = art_wd(p,q)
        sum_of.append(np.mean(wd))
    
    # print(sum_of)
    
    return np.mean(sum_of)

def kl_divergence(samples: torch.Tensor,target: torch.Tensor) -> torch.Tensor: 
    """KL Divergence Estimation

    Args:
        samples (Tensor): Output disitrbution
        target (Tensor): Target distribution

    Returns:
        Tensor: KLD between target and samples
    """    
    sum_of=[]
    
    if len(target) < 2:
        q = target[0]
        for p in samples:
            # test = sum([p[i] * torch.log(p[i]/q[i]) for i in range(len(p))])
            i = q.argmax()
            test = p[i] * torch.log(p[i]/q[i])
            sum_of.append(test.cpu().detach())
    else:
        for p, q in zip(samples,target):
            # test = sum([p[i] * torch.log(p[i]/q[i]) for i in range(len(p))])
            i = q.argmax()
            test = p[i] * torch.log(p[i]/q[i])
            sum_of.append(test.cpu().detach())
        
    try:
        return np.mean(sum_of)
    except: 
        return torch.tensor(0.005)

def logits_score(output, target):
    sum_of = []
    for idx, out in enumerate(output):
        scores, index = torch.topk(out, 2)
        sum_of.append(torch.norm(scores[0] - scores[1], 2).detach().cpu().numpy())

    return np.mean(sum_of)
    
def score_bo(candidates: torch.Tensor, target:float, uncertainty, model: torch.Tensor, config: dict):
    """
    Scoring the Bayesian optimisation and forming what it will go into GP
    """
    weights, _, _ = read_weights(model, config)
    scores = None
    
    for idx, candidate in enumerate(candidates):
        weights = weights.to(candidate.device)
        
        if idx > 1:
            scores = torch.cat(scores, torch.norm(weights - candidate, p=2))  
        else:
            scores = torch.norm(weights - candidate, p=2)
            scores = scores.reshape(1, scores.shape[0])
    
    score, idx = torch.kthvalue(scores, 0)
    
    return False, _ if score > target else True, idx

def get_uncertainty(model, x, y, config):
    if not(len(x) > 0):
        return 10
    device =  next(model.parameters()).device
    model.eval()
    x, y = x.to(device), y.to(device)

    logits = model(x)
    output = F.softmax(logits, dim=1)
        
    if config["attack"]["objective"] == "kl_div":
        target = confidence_level(y, config["attack"]["target_confidence"], output.shape[1])
        score = kl_divergence(output, target)
        
        if config["attack"]["use_logits"]:
            score = logits_score(logits, y)
    
    elif config["attack"]["objective"] == "mutual_info":
        score = mi_BO(output, target)  
        
    elif config["attack"]["objective"] == "wasserstein":
        score = torch.Tensor([wasserstein_matrix(output, target)])
        
    elif config["attack"]["objective"] == "js_div":
        target = confidence_level(y, 0, output.shape[1])
        
        score = np.mean(jensenshannon(output.detach().cpu().numpy(), target.detach().cpu().numpy(), axis=1))
    
    return score

def get_true_positives(output, target):
    best_samples, best_target = [], []
    for i in range(len(output)): 
        if torch.argmax(output[i]) == target[i]:
            best_samples.append(output[i])
            best_target.append(target[i])
            
    return best_samples, best_target