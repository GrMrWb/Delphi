import copy
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.delphi.utils.weight_io import read_weights, write_weights

logger = logging.getLogger(__name__)

def modify_the_weights_with_single_neuron(model: torch.Tensor, lambda_samples: torch.Tensor, device, best_feature_idx, explainable : bool = False, **kwargs) -> torch.Tensor:
    model = model.to(device)
    model_list = copy.deepcopy(list(model.parameters()))
    new_model = copy.deepcopy(model)
    
    if "layer" in kwargs:
        layer = kwargs["layer"] if isinstance(kwargs["layer"], list) else [kwargs["layer"]]
    
    if "server_config" in kwargs:
        server_config=kwargs["server_config"]
    
    weights, grads, type_of_layer = read_weights(new_model, server_config)
        
    new_weights = copy.deepcopy(weights)
    
    lambda_samples = lambda_samples.reshape(new_weights[layer[0]][int(best_feature_idx)].shape).to(dtype=new_weights[layer[0]].dtype, device=new_weights[layer[0]].device)

    new_weights[layer[0]][int(best_feature_idx)] = lambda_samples
    
    return write_weights(model, new_weights, server_config, layer)
  