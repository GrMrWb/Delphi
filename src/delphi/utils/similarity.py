import numpy as np

def get_similarity_features(data_points):
    """_summary_

    Args:
        data_points (_type_): _description_

    Returns:
        numpy.ndarray: cosine similarity of the features
    """
    similarity=[]
    
    for data in data_points:
        best_value = data_points[0][0][0].cpu().detach().numpy()
        check_value = data[0][0].cpu().detach().numpy()
        result = np.sum(best_value*check_value)/(np.linalg.norm(best_value)*np.linalg.norm(check_value))
        similarity.append(result)
        
    return np.array(similarity).reshape(-1,1)

def get_similarity_between_server(weight_before, weight_after) -> np.array:
    """_summary_

    Args:
        weight_before (_type_): _description_
        weight_after (_type_): _description_

    Returns:
        np.array: _description_
    """
    
    similarity=[]
    
    try:
        for bef_opt, aft_opt in zip(weight_before, weight_after):
            A = bef_opt[0][0].cpu().detach().numpy()
            B = aft_opt[0][0].cpu().detach().numpy()
            result = np.sum(A*B)/(np.linalg.norm(A)*np.linalg.norm(B))
            similarity.append(result)
    except:
        for bef_opt, aft_opt in zip(weight_before, weight_after):
            A = bef_opt.cpu().detach().numpy()
            B = aft_opt.cpu().detach().numpy()
            result = np.sum(A*B)/(np.linalg.norm(A)*np.linalg.norm(B))
            similarity.append(result)
        
        
    return np.array(similarity).mean()