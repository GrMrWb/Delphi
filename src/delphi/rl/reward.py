import numpy as np
import copy
import torch
import torch.nn as nn

from src.delphi.utils import (
    write_weights, 
    read_weights, 
    get_uncertainty, 
    modify_the_weights_with_single_neuron
)

def calculate_reward(model, data, target, new_weights, indices, device, config):
    uncertainties = []
    counter=0
    rewards = []
    new_model = copy.deepcopy(model)
    
    for index in indices:
        new_model = modify_the_weights_with_single_neuron(new_model, new_weights[counter], device, index, server_config=config)
        counter+=1
    
    final =  float(get_uncertainty(new_model, data, target, config))
    counter=0
    
    overall = get_uncertainty(model, data, target, config)
    
    for index in indices:
        reward = 0
        new_model = modify_the_weights_with_single_neuron(model, new_weights[counter], device, index, server_config=config)
        
        delta = final - float(get_uncertainty(new_model, data, target, config))
        
        reward = 1+delta if delta > 0 else  delta-1
        
        rewards.append(reward)
        counter+=1
        
    return torch.tensor(rewards), final

def calculate_single_reward(model, data, target, new_weights, indices, device, config):
    new_model = copy.deepcopy(model)
    weights, _ ,_ = read_weights(model, config)

    candidates = []
    begin = 0
    for layer, values in indices.items():
        for index in values:
            end = begin + np.array(weights[layer][index].shape).prod()
            candidates.append(new_weights[:, begin:end])
            begin = end

    layers = config["attack"]["layer_of_interest"] if isinstance(config["attack"]["layer_of_interest"], list) else [config["attack"]["layer_of_interest"]]
    counter=0
    
    for layer, values in indices.items():
        for index in values:
            new_model = modify_the_weights_with_single_neuron(copy.deepcopy(new_model), candidates[counter], device, index, layer=layer, server_config=config)
            counter+=1
    
    new_model = copy.deepcopy(new_model)
    
    final =  float(get_uncertainty(new_model, data, target, config))
    counter=0
    
    overall = get_uncertainty(model, data, target, config)

    delta = final - float(get_uncertainty(model, data, target, config))
    
    reward = -np.log(final) if final>0 else final
        
    return reward, final