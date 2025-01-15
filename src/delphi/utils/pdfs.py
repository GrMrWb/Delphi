import numpy as np
import torch

def pdf_for_output(outputs) -> np.array:
    
    sumOfGrad = np.zeros((outputs.shape[1]))
    deviations = np.zeros((outputs.shape[1]))
    
    for i, num in enumerate(outputs):
        for idx, output in enumerate(num):
            sumOfGrad[idx] += output.sum()/(output.shape[0]*output.shape[1])
        
    averages = sumOfGrad / outputs.shape[0]
    
    for i, num in enumerate(outputs):
        for idx, output in enumerate(num):
            deviations[idx] += (output.sum() - averages[idx])**2
    
    deviations = deviations / outputs.shape[0]
    
    return {'deviation': deviations, 'average': averages}

def pdf_for_outputs(new_outputs, old_outputs, device) -> np.array:
    
    new_sumOfGrad = torch.zeros((new_outputs.shape[1])).to(device)
    new_deviations = torch.zeros((new_outputs.shape[1])).to(device)
    
    old_sumOfGrad = torch.zeros((old_outputs.shape[1])).to(device)
    old_deviations = torch.zeros((old_outputs.shape[1])).to(device)
    
    for i, (new, old) in enumerate(zip(new_outputs, old_outputs)):
        for idx, output in enumerate(new):
            new_sumOfGrad[idx] += output.sum()/(output.shape[0]*output.shape[1])
            old_sumOfGrad[idx] += old[idx].sum()/( old[idx].shape[0]* old[idx].shape[1])
        
    new_averages = new_sumOfGrad / new_outputs.shape[0]
    old_averages = old_sumOfGrad / old_outputs.shape[0]
    
    for i, (new, old) in enumerate(zip(new_outputs, old_outputs)):
        for idx, output in enumerate(new):
            new_deviations[idx] += (output.sum() - new_averages[idx])**2
            old_deviations[idx] += (old[idx].sum() - old_averages[idx])**2
            
    new_deviations = new_deviations / new_outputs.shape[0]
    old_deviations = old_deviations / old_outputs.shape[0]
    
    return {'deviation': new_deviations, 'average': new_averages}, {'deviation': old_deviations, 'average': old_averages}
