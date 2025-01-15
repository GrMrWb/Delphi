import numpy as np

def get_weight_distribution(model, best_features, weights=False):
    if not weights:
        try:
            features = model if weights else model.net.conv1.weight.data
        except:
            features = model
            
        mean, deviation = np.zeros(len(best_features)), np.zeros(len(best_features))
        
        count = 0
        for bf_idx in best_features:
            mean[count] = features[bf_idx].sum()
            count+=1
        
        mean /= (features.shape[2]*features.shape[3])
        
        count = 0
        for bf_idx in best_features:
            deviation[count] = (features[bf_idx].sum() - mean[count])**2
            count+=1
            
        deviation /= (features.shape[2]*features.shape[3])
        
        return np.concatenate((mean.reshape(-1, 1), deviation.reshape(-1, 1)), axis=1)
    
    else:
        mean= model[0][0].cpu().detach().numpy().mean()
        deviation = np.std(model[0][0].cpu().detach().numpy())
        
        return mean, deviation