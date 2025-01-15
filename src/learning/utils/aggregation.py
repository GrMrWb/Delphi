import torch
import torch.nn.functional as F
import numpy as np
import copy

def Multi_Krum(dict_parameters, previous_weight):
    """
    multi krum passed parameters.
    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """
    from torch.nn import functional as F
    import numpy as np

    multi_krum = 5
    candidate_num = 7
    distances = {}
    tmp_parameters = {}
    pre_distance = []
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((_parameter[name].data - parameter[name].data).float()) for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
            
        distance.sort()
        distances[idx] = sum(distance[:candidate_num+1])

    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:multi_krum]
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params

def Krum(dict_parameters, num_clients) -> torch.Tensor:
    """
    krum passed parameters.
    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """

    candidate_num = 6
    distances = {}
    tmp_parameters = {}
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        
        distance.sort()
        
        # print("benign distance: " + str(distance))
        distances[idx] = sum(distance[:candidate_num])
    
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:1]

    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params

def Multi_Krum_Cuda(dict_parameters, previous_weight):
    """
    multi krum passed parameters.
    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """
    from torch.nn import functional as F
    import numpy as np

    multi_krum = 5
    candidate_num = 7
    distances = {}
    tmp_parameters = {}
    pre_distance = []
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((_parameter[name].data.to('cuda:0') - parameter[name].data.to('cuda:0')).float().to('cuda:0')) for name in parameter.keys()]
            distance.append(torch.sum(dis))
            tmp_parameters[idx] = parameter
            
        distance.sort()
        distances[idx] = torch.sum(distance[:candidate_num+1])

    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:multi_krum]
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = torch.sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params

def KrumCuda(dict_parameters) -> torch.Tensor:
    """
    krum passed parameters.
    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """

    candidate_num = len(dict_parameters)
    distances = {}
    tmp_parameters = {}
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = torch.tensor([torch.norm((parameter[name].data.to('cuda:0') - _parameter[name].data.to('cuda:0')).float().to('cuda:0'))**2 for name in parameter.keys()])
            distance.append(torch.sum(dis))
            tmp_parameters[idx] = parameter
        
        distance.sort()
        
        # print("benign distance: " + str(distance))
        distances[idx] = torch.sum(torch.tensor(distance[:candidate_num]).to('cuda:0'))
    
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:1]
    

    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += copy.deepcopy(w[i][k])
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def PFedAvg(model, weights):
    # idxs = [0, 1, 5, 6]
            
    global_weights = FedAvg(weights)
    prev_model = copy.deepcopy(model)
    
    model.load_state_dict(global_weights)
    
    beta = 0.95
    # aaggregate avergage model with previous model using parameter beta
    for pre_param, param in zip(prev_model.parameters(), model.parameters()):
        param.data = (1 - beta)*pre_param.data + beta*param.data
        
    return model.state_dict()