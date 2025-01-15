import numpy as np
import copy
import logging
import torch
import torch.nn.functional as F

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

# from src.delphi.attacks import *
# from src.delphi.analyser import *
from src.delphi.convex_opt import LeastSquareModelOptimiser
from src.delphi.rl.utils import run_single_ppo

from src.delphi.ubo import *
from src.delphi.utils import (
    get_uncertainty, find_the_best_features_FL,
    modify_the_weights_with_single_neuron, read_weights,
    get_historical_data, get_historical_gradients, get_historical_distribution
    )

from importlib import reload

reload(logging)
logger = logging.getLogger(__name__)

def apply_les(config: dict, model: torch.Tensor, testset, device, **kwargs):
    modified_model = copy.deepcopy(model)
    objective = config["attack"]["objective"]
    
    sample_data, sample_target = [], []

    """Task 1: Get a good sample for analysis"""
    model = model.to('cpu')
    for data, target in testset:
        output = F.softmax(model(data), dim=1)
        for i in range(len(output)):
            if torch.argmax(output[i]) == target[i] and target[i] in config["attack"]["target_class"]:
                sample_data.append(data[i].detach().cpu().numpy())
                sample_target.append(int(target[i].detach().cpu().float()))
                
            if len(sample_target) > 50: 
                break
    
    sample_data, sample_target = torch.tensor(sample_data), torch.tensor(sample_target)
    
    if len(sample_target) < 1:
        analysis = {
            "lambda_sample" : 0,
            "score" : 0,
        }
        return False, analysis
    
    """Task 2: Get the uncertainty of the model"""
    curr_uncertainty = get_uncertainty(model, sample_data, sample_target, config)
    
    """Task 3: Set the aim of the uncertainty"""
    target_objective = curr_uncertainty * (1 - config["attack"]["uncertainty"]["uncertainty_bounds"])
    config["attack"]["objectives"][f"{objective}_target"] = 0.5 if curr_uncertainty > 1 else target_objective
    config["attack"]["objectives"][f"{objective}_target"] = 0 if curr_uncertainty  < 0.5 else target_objective
        
    print(f"Target: {config['attack']['objectives'][f'{objective}_target']:.5f} \n Current Uncertainty: {curr_uncertainty:.5f}")
    
    """Task 4: Calculate the best features"""
    best_features_indices = find_the_best_features_FL(model, sample_data, sample_target, server_config=config) if config["attack"]["best_features"] != "fixed" else kwargs["best_indices"]
    
    """Task 5: Amount of bounds to be tested"""
    #TODO add a function do choose the boundaries
    experiments_bounds = np.logspace(-1, 0, config["attack"]["least_squares"]["number_of_explorations"], endpoint=True)
    idx = 0
    best_features_idx = best_features_indices[idx]
    samples = {}
    
    """Task 6: Get the objective"""
    if config["attack"]["type_of_optimisation"] == "least_squares":
        library = LeastSquareModelOptimiser(sample_data, sample_target, modified_model, config, best_features_idx, kl_div=True)
    
    optimise = True
    trial = 0
    while optimise:
        try:
            trial+=1
            res = library.run()
            print(f"Function: {res.fun}")
            lambda_sample = res.x
            modified_model = modify_the_weights_with_single_neuron(modified_model, torch.tensor(lambda_sample), device, best_features_idx, server_config=config)
            
            if library.obj_diff > 0.05:
                if config["attack"]["least_squares"]["continue"]:
                    idx += 1
                    if idx > (len(best_features_indices)-1):
                        optimise = False
                    else:
                        best_features_idx = best_features_indices[idx]
                        print(f"Change of index, We use index: {best_features_idx}")
                        library.feature_idx = best_features_idx
                        library.neural_net = copy.deepcopy(modified_model)                
                else:
                    optimise = False
            else:
                optimise= False
            
        except:
            print(f"Working on {library.obj_diff}")
            lambda_sample = library.last_x
            modified_model = modify_the_weights_with_single_neuron(modified_model, torch.tensor(lambda_sample), device, best_features_idx, server_config=config)
            samples[f"{best_features_idx}"] = lambda_sample
            
            if config["attack"]["least_squares"]["continue"]:
                idx += 1
                if idx > (len(best_features_indices)-1):
                    optimise = False
                else:
                    best_features_idx = best_features_indices[idx]
                    print(f"Change of index, We use index: {best_features_idx}")
                    library.feature_idx = best_features_idx
                    library.neural_net = copy.deepcopy(modified_model)                
            else:
                optimise = False
                
            if trial > 5:
                optimise = False
        
        
        samples[f"{best_features_idx}"] = lambda_sample
        
    analysis = {
        "lambda_sample" : samples,
        "score" : library.obj_diff[0],
    }
    
    return modified_model, analysis

def apply_usobo(config: dict, model: torch.Tensor, testset, device, historical_data, **kwargs):
    modified_model = copy.deepcopy(model)
    objective = config["attack"]["objective"]
    
    sample_data, sample_target = [], []

    """Task 1: Get a good sample for analysis"""
    model = model.to(device)
    for data, target in testset:
        data, target = data.to(device), target.to(device)
        output = F.softmax(model(data), dim=1)
        for i in range(len(output)):
            if torch.argmax(output[i]) == target[i]:# and target[i] in config["attack"]["target_class"]:
                sample_data.append(data[i].detach().cpu().numpy())
                sample_target.append(int(target[i].detach().cpu().float()))
            
            if len(sample_target) > 128:
                break
    
    sample_data, sample_target = torch.tensor(sample_data), torch.tensor(sample_target)
    
    """Task 2: Get the uncertainty of the model"""
    curr_uncertainty = get_uncertainty(model, sample_data, sample_target, config)
    
    """Task 3: Set the aim of the uncertainty"""
    target_objective = curr_uncertainty * (1 - config["attack"]["uncertainty"]["uncertainty_bounds"])
    
    config["attack"]["objectives"][f"{objective}_target"] = 0.5 if curr_uncertainty > 1 else target_objective
    config["attack"]["objectives"][f"{objective}_target"] = 0 if curr_uncertainty  < 0.5 else target_objective
    config["attack"]["objectives"][f"{objective}_target"] = 0
    
    print(f"Target: {config['attack']['objectives'][f'{objective}_target']:.5f} \n Current Uncertainty: {curr_uncertainty:.5f}")
    
    """Task 4: Calculate the best features"""
    
    best_features_indices = find_the_best_features_FL(model, sample_data, sample_target, server_config=config) if config["attack"]["best_features"] != "fixed" else kwargs["best_indices"][-1]
    
    """Task 5: Prepare the GP for the dataset"""
    train_y = torch.tensor(historical_data["uncertainty"])
    train_set_x = {}
    for data in range(len(historical_data["weights"])):
        for layer, indices in best_features_indices.items():
            for key, index in enumerate(indices):
                if f"{layer}" not in train_set_x:
                    train_set_x[layer] = {f"{index}" : [historical_data['weights'][data][layer][key]]} 
                elif f"{index}" not in train_set_x[layer]:
                    train_set_x[layer][f"{index}"] = [historical_data['weights'][data][layer][key]]
                else:
                    train_set_x[layer][f"{index}"].append(historical_data['weights'][data][layer][key])
            
    for layer, indices in best_features_indices.items():
        for key, index in enumerate(indices):
            train_set_x[layer][f"{index}"] = torch.tensor(train_set_x[layer][f"{index}"])
    
    """Task 6: Create the GPs for the historical data"""
    model_list_gp = {}
    
    for layer, indices in best_features_indices.items():
        for key, index in enumerate(indices):
            model_gp = SingleTaskGP(train_set_x[layer][f"{index}"], train_y.view(-1,1))
            mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
            
            from botorch import fit_gpytorch_model
            fit_gpytorch_model(mll)

            if f"{layer}" not in model_list_gp:
                model_list_gp[layer] = {f"{index}" : model_gp }
            else:
                model_list_gp[layer][f"{index}"] = model_gp
    
    """Task 7: Run the algorithm"""
    weight_optimizer = USOBO_FLPA(config, best_features_indices, model, sample_data, sample_target, model_list_gp)
    
    modified_model, scores, lambdas, model_gp = weight_optimizer.time_for_magic(runs=20)
    
    # data_gp = {
    #     "scores": weight_optimizer.scores,
    #     "lambdas": weight_optimizer.scores,
    # }

    analysis = {
        "lambda_sample" : lambdas,
        "score" : scores,
        "best_indice" : best_features_indices,
    }
    
    del weight_optimizer
    
    return modified_model, analysis

def apply_umobo(config: dict, model: torch.Tensor, testset, device, historical_data, **kwargs):
    modified_model = copy.deepcopy(model)
    objective = config["attack"]["objective"]
    
    sample_data, sample_target = [], []

    """Task 1: Get a good sample for analysis"""
    model = model.to(device)
    for data, target in testset:
        data, target = data.to(device), target.to(device)
        output = F.softmax(model(data), dim=1)
        for i in range(len(output)):
            if torch.argmax(output[i]) == target[i]:# and target[i] in config["attack"]["target_class"]:
                sample_data.append(data[i].detach().cpu().numpy())
                sample_target.append(int(target[i].detach().cpu().float()))
            
            if len(sample_target) > 128:
                break
    
    sample_data, sample_target = torch.tensor(sample_data), torch.tensor(sample_target)
    
    """Task 2: Get the uncertainty of the model"""
    curr_uncertainty = get_uncertainty(model, sample_data, sample_target, config)
    
    """Task 3: Set the aim of the uncertainty"""
    target_objective = curr_uncertainty * (1 - config["attack"]["uncertainty"]["uncertainty_bounds"])
    
    config["attack"]["objectives"][f"{objective}_target"] = 0.5 if curr_uncertainty > 1 else target_objective
    config["attack"]["objectives"][f"{objective}_target"] = 0 if curr_uncertainty  < 0.5 else target_objective
    config["attack"]["objectives"][f"{objective}_target"] = 0
    
    print(f"Target: {config['attack']['objectives'][f'{objective}_target']:.5f} \n Current Uncertainty: {curr_uncertainty:.5f}")
    
    """Task 4: Calculate the best features"""
    
    best_features_indices = find_the_best_features_FL(model, sample_data, sample_target, server_config=config) if config["attack"]["best_features"] != "fixed" else kwargs["best_indices"][-1]
    
    """Task 5: Run the algorithms"""
    
    weight_optimizer = UMOBO_FLPA(config, best_features_indices, model, sample_data, sample_target)
    
    modified_model, scores, lambdas, model_gp = weight_optimizer.time_for_magic(runs=20)

    analysis = {
        "lambda_sample" : lambdas,
        "score" : scores,
        "best_indice" : best_features_indices,
    }
    
    del weight_optimizer
    
    return modified_model, analysis

def apply_rl(config: dict, model: torch.Tensor, testset, device, **kwargs):
    modified_model = copy.deepcopy(model)
    objective = config["attack"]["objective"]
    
    sample_data, sample_target = [], []

    """Task 1: Get a good sample for analysis"""
    model = model.to(device)
    for data, target in testset:
        data, target = data.to(device), target.to(device)
        output = F.softmax(model(data), dim=1)
        for i in range(len(output)):
            if torch.argmax(output[i]) == target[i]:# and target[i] in config["attack"]["target_class"]:
                sample_data.append(data[i].detach().cpu().numpy())
                sample_target.append(int(target[i].detach().cpu().float()))
            
            if len(sample_target) > 128:
                break
    
    sample_data, sample_target = torch.tensor(sample_data), torch.tensor(sample_target)
    
    """Task 2: Get the uncertainty of the model"""
    curr_uncertainty = get_uncertainty(model, sample_data, sample_target, config)
    
    """Task 3: Set the aim of the uncertainty"""
    config["attack"]["objectives"][f"{objective}_target"] = 0
     
    print(f"Target: {config['attack']['objectives'][f'{objective}_target']:.5f} \n Current Uncertainty: {curr_uncertainty:.5f}")
    
    """Task 4: Calculate the best features"""
    best_features_indices = find_the_best_features_FL(model, sample_data, sample_target, server_config=config) if config["attack"]["best_features"] != "fixed" else kwargs["best_indices"]
    
    """Task 5: Run the algorithm"""
    model, uncertainty = run_single_ppo(config, model, best_features_indices, sample_data, sample_target)
    
    return model, {"score": uncertainty, "best_indice" : best_features_indices}