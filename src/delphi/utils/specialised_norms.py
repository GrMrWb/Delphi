import numpy as np
import torch

def l2_norm(target_mean, target_deviation, distributions, one=False):
    if not one:
        target_mean_l2 = np.zeros(len(distributions))
        target_deviation_l2 = np.zeros(len(distributions))
        count=0
    
        for distr in distributions:
            target_mean_l2[count] = np.sqrt(distr[0]**2 + target_mean**2)
            target_deviation_l2[count] = np.sqrt(distr[0]**2 + target_mean**2)
            count+=1
    else:
        target_mean_l2, target_deviation_l2 = 0 , 0
    
        target_mean_l2 = np.sqrt(distributions[0]**2 - target_mean**2)
        target_deviation_l2 = np.sqrt(distributions[1]**2 - target_deviation**2)
    
    # print(target_mean_l2, target_deviation_l2)
    
    return target_mean_l2, target_deviation_l2
    
def get_delta_weights(A, B):
    A, B = A.to(A.device), B.reshape(B.shape[1]).to(A.device)
    return torch.norm(A - B, p=1)