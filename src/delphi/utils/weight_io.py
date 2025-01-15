import torch.nn as nn
import re

def remove_digits(s):
    return re.sub(r'\d+', '', s)

def read_weights(model, server_config):
    configuration = server_config
    
    layers = configuration["attack"]["layer_of_interest"]
    strings = ["bn", "norm", "attn", "bias", "reduction"]
    
    multi_modality = False if server_config["learning_config"]["modality"] == "single" else True
    modal = f'{server_config["attack"]["modality"]}_encoder'
    
    layers = layers if isinstance(layers, list) else [layers]
        
    parameters = list(model.named_parameters())
    layers_named_weights = []
    type_of_layer = []
    
    for parameter in parameters:
        name = parameter[0].split(".")
        word_count = sum(
            1 for word in name 
            if any(remove_digits(word) == remove_digits(s) for s in strings)
        )
        
        if word_count==0 and "weight" in name:
            if len(parameter[1].shape) > 1:
                layers_named_weights.append(parameter)      
        
        if multi_modality:
            if not modal in name:
                layers_named_weights.pop()
        
    weights, gradients = {}, {}
    
    for layer in layers:
        index = layer-1
        name_list = layers_named_weights[index][0].split(".")
            
        temp = model
        for name in name_list:
            temp = getattr(temp, name)
            
        weights[str(index)]  = temp.data
        gradients[str(index)]  = temp.grad.data if temp.grad is not None else 0
        
        type_of_layer.append(layers_named_weights[index][0])
    
    return weights, gradients, type_of_layer

def write_weights(model, new_weights, server_config, layer):
    configuration = server_config
    
    strings = ["bn", "norm", "attn", "bias", "reduction" ]
    
    multi_modality = False if server_config["learning_config"]["modality"] == "single" else True
    modal = f'{server_config["attack"]["modality"]}_encoder'
        
    parameters = list(model.named_parameters())
    layers_named_weights = []
    
    for parameter in parameters:
        name = parameter[0].split(".")
        word_count = sum(
            1 for word in name 
            if any(remove_digits(word) == remove_digits(s) for s in strings)
        )
        
        if word_count==0 and "weight" in name:
            if len(parameter[1].shape) > 1:
                layers_named_weights.append(parameter)   
            
        if multi_modality:
            if not modal in name:
                layers_named_weights.pop()
                
    index = int(layer[0])
    
    name_list = layers_named_weights[index][0].split(".")
    
    temp = model
    for name in name_list:
        temp = getattr(temp, name)
    
    temp.data = new_weights[str(index)]
    
    return model
