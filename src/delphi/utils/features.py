import torch
import numpy as np
from src.learning.utils import CustomCrossEntropy
from src.delphi.utils.weight_io import read_weights

def find_the_best_features(target, model, old_weights) -> np.array:
    """This is for SL

    Args:
        target (_type_): _description_
        model (_type_): _description_
        old_weights (_type_): _description_

    Returns:
        np.array: _description_
    """
    gradients = model.net.conv1.weight.grad.data
    weights = model.net.conv1.weight.data

    sum_of_gradients, best_gradients, best_weights, best_old_weights = [], [], [], []
    
    for gradient in gradients:
        sum_of_gradients.append(gradient.abs().sum())
    
    sum_of_gradients = torch.tensor(sum_of_gradients)
    
    features = []
    count=0;
    
    try:
        while count < 5:
            index = (sum_of_gradients == sum_of_gradients.max()).nonzero()
            features.append(index.item())
            best_gradients.append(gradients[index][0])
            best_weights.append(weights[index][0])
            best_old_weights.append(old_weights[index][0])
            sum_of_gradients[index] = 0
            count+=1
    except:
        features = 0
    
    return np.array(features), best_gradients, best_weights, best_old_weights

def find_the_best_features_FL(model, data, target, **kwargs):
    #TODO modify this for multiple neurons and layers
    """This is for FL

    Args:
        model (_type_): _description_
        data (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """
    if "server_config" in kwargs:
        server_config = kwargs["server_config"]
    
    device =  next(model.parameters()).device
    loss_function = CustomCrossEntropy(target_confidence=0.25)
    
    data, target = data.to(device), target.to(device)
                
    output = model(data)
    
    loss_func = loss_function(output, target)
    loss_func.backward()
    
    if isinstance(model, list):
        gradients = model[0].weight.grad.data
        weights = model[0].weight.data
    else:
        weights, gradients, type_of_layer = read_weights(model, server_config)
           
    top_k_indices = {}         
    for key, gradient in gradients.items():
        sum_of_gradients = []
        
        for grads in gradient:
            sum_of_gradients.append(torch.norm(grads))
        
        sum_of_gradients = torch.tensor(sum_of_gradients)
        
        top_k_indices[key] = sum_of_gradients.topk(server_config["attack"]["num_of_neurons"])[1]
    
    return top_k_indices

def get_bounds(weights, config, numerical=False):
    up = config["attack"]["bounds"]
    dn = -1*config["attack"]["bounds"]
    upper_bound = []
    lower_bound = []
    
    if len(weights.shape) > 1:
        upper_bound = torch.zeros_like(weights)
        lower_bound = torch.zeros_like(weights)
        
        if not numerical:
            for dim_idx, dimension in enumerate(weights):
                for row_idx, row in enumerate(dimension):
                    for col_idx, column in enumerate(row):
                        if column > 0:
                            upper_bound[dim_idx][row_idx][col_idx] = column*up
                            lower_bound[dim_idx][row_idx][col_idx] = column*dn
                        elif column < 0:
                            lower_bound[dim_idx][row_idx][col_idx] = column*up
                            upper_bound[dim_idx][row_idx][col_idx] = column*dn
                        else:
                            upper_bound[dim_idx][row_idx][col_idx] = dimension.mean()*up
                            lower_bound[dim_idx][row_idx][col_idx] = dimension.mean()*dn
                            
        else:
            upper_bound = torch.add(upper_bound, -1*up)
            lower_bound = torch.sub(lower_bound, -1*dn)
            
        bounds = torch.cat([
            lower_bound.reshape(1, lower_bound.shape[0],lower_bound.shape[1], lower_bound.shape[2]),
            upper_bound.reshape(1, lower_bound.shape[0],lower_bound.shape[1], lower_bound.shape[2])
        ])

    else:
        for weight in weights:
            if not numerical:
                if weight > 0:
                    
                    lower_bound.append(weight*dn)
                    upper_bound.append(weight*up)
                else:
                    lower_bound.append(weight*up)
                    upper_bound.append(weight*dn)
            else:
                lower_bound.append(dn)
                upper_bound.append(up)
        
        bounds = torch.tensor([lower_bound, upper_bound])
    
    bounds = bounds.to(dtype=torch.float32)
    
    return bounds