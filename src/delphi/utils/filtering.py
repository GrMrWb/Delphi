import torch
import numpy as np

def get_the_best_samples(output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Create a tensor array for the best sample which have confidene
    greater than 0.75 on the target class

    Args:
        output (Tensor): predicted values from the model
        y (Tensor): actual values

    Returns:
        Tensor: best output and actual values
    """    
    best_samples = []
    best_samples_target = []

    for index, sample in enumerate(output):
        if sample.max() > 0.25:
            best_samples.append(sample.detach().cpu().numpy())
            best_samples_target.append(y[index].item())
            
    if len(best_samples) == 0:
        for index, sample in enumerate(output):
            if sample.max() > 0.12:
                best_samples.append(sample.detach().cpu().numpy())
                best_samples_target.append(y[index].item())
    
    if len(best_samples) == 0:
        best_samples = torch.tensor([0])
        best_samples_target = torch.tensor([0])
    else:
        best_samples = torch.tensor(best_samples)
        best_samples_target = torch.tensor(best_samples_target)
    
    return best_samples, best_samples_target

def filter_data_for_BO(weights, similarities):
    """
    TODO: create a function which will filter dataset for the Bayesian Optimisation, ideally around 100 samples
    """
    indice_remove = []
    
    for idx, similarity in enumerate(similarities):
        if similarity > 0.98:
            indice_remove.append(idx)
    
    weights = np.delete(weights, np.array(indice_remove), axis=0)
    similarities = np.delete(similarities, np.array(indice_remove), axis=0)
    
    return weights, similarities

def filter_output(testset, model, server_config):
    sample_data = []
    sample_target = []
    for data, target in testset:
        model = model.to('cpu')
        output = F.softmax(model(data), dim=1)
        for i in range(len(output)):
            if output[i].max() > server_config["attack"]["filtering"]["confidence"] and torch.argmax(output[i]) == target[i] and target[i] in server_config["attack"]["target_class"]:
                sample_data.append(data[i].detach().cpu().numpy())
                sample_target.append(int(target[i].detach().cpu().float()))
                
            if len(sample_target)>50: 
                break
        
    sample_data = torch.tensor(sample_data)
    sample_target = torch.tensor(sample_target)
    
    return sample_data, sample_target