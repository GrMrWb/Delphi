import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.delphi.utils.measurements import get_uncertainty, logits_score
from src.delphi.utils.weight_io import read_weights

def get_historical_data(model, dataset, server_config):
        
    sample_data, sample_target = [], []
    sample_logits = []
    
    model.eval()
    model = model.to(server_config["device"])
    for data, target in dataset:
        try:
            data, target = data.to(server_config["device"]), target.to(server_config["device"])
            
            logits = model(data)
            output = F.softmax(logits, dim=1)
            
            for i in range(len(output)):
                if torch.argmax(output[i]) == target[i] and target[i] in server_config["attack"]["target_class"]:
                    sample_data.append(data[i].detach().cpu().numpy())
                    sample_logits.append(logits[i].detach().cpu().numpy())
                    sample_target.append(int(target[i].detach().cpu().float()))
                    
                if len(sample_target) > 250: 
                    break
        except:
            pass
    
    sample_data, sample_target = torch.tensor(sample_data), torch.tensor(sample_target)
    sample_logits = torch.tensor(sample_logits)
    try:
        uncertainty = get_uncertainty(model, sample_data, sample_target, server_config)
        
        if server_config["attack"]["use_logits"]:
            uncertainty = logits_score(sample_logits, sample_target)
            
    except:
        uncertainty = torch.tensor(5.)

    weights, _, _ = read_weights(model, server_config)
    
    return uncertainty, weights

def get_historical_gradients(model, dataset, server_config):
    model.eval()
    model = model.to(server_config["device"])
    sample_target, sample_data = [], []
    loss_func = nn.CrossEntropyLoss()
    for data, target in dataset:
        data, target = data.to(server_config["device"]), target.to(server_config["device"])
        output = F.softmax(model(data), dim=1)
        for i in range(len(output)):
            if torch.argmax(output[i]) == target[i]:
                sample_data.append(data[i].detach().cpu().numpy())
                sample_target.append(int(target[i].detach().cpu().float()))
                
                if len(sample_target) > 250: 
                    break
    
    sample_data, sample_target = torch.tensor(sample_data, device=server_config["device"]), torch.tensor(sample_target, device=server_config["device"])
    # try:
    output = model(sample_data)
    loss = loss_func(output, sample_target)
    loss.backward()
    
    _, gradients, _ = read_weights(model, server_config)

    norm_gradients = []
    
    for key, gradient in gradients.items():
        norm_gradients.append(torch.norm(torch.tensor(gradient)))

    norm_gradients = torch.tensor(norm_gradients).flatten().detach().cpu().numpy()
    
    del model, dataset, sample_data, sample_target
    
    return norm_gradients

def get_historical_distribution(model, server_config):
    weights, _, _ = read_weights(model, server_config)
    
    layer = server_config["attack"]["layer_of_interest"] if isinstance(server_config["attack"]["layer_of_interest"], list) else [server_config["attack"]["layer_of_interest"]]

    mean, std = weights[str(layer[0]-1)].mean(dim=(1,2,3)).flatten(), weights[str(layer[0]-1)].std(dim=(1,2,3)).flatten()
        
    mean, std = mean.detach().cpu().numpy(), std.detach().cpu().numpy()
        
    return mean, std