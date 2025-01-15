import numpy as np
import torch
import torch.nn as nn

from src.delphi.utils.weight_io import read_weights

def get_activations(model, server_config):
    configuration = server_config
    model_name = configuration["learning_config"]["global_model"]["type"]
    if model_name == "AlexNet" :
        if hasattr(model, "net"):
            weights = getattr(model.net, "conv1")
    elif model_name == "ExplainableAlexNet":
        pass
    elif model_name[:4] == "MLP_": # Model: MLP_MNIST and MLP_CIFAR
        if hasattr(model, "network"):
            weights = getattr(model, "network")[1]
        else:                
            weights = getattr(model, "fc1")
        
    return weights, model_name

def capture_data(inputs, model, server_config):
    layer, model_name = get_activations(model, server_config)
    weights, _, _ = read_weights(model, server_config)
    flat = nn.Flatten()
    act =  nn.ReLU()
    x = flat(inputs) if model_name[:4]=="MLP_" else inputs.clone()

    activation = act(layer(x))

    return weights, activation