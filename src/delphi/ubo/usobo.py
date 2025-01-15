import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon

from src.delphi.utils import *
from src.delphi.utils.bo_libraries import *

class LayerOptUSOBO:
    def __init__(self, config, indices, model, data, target, ground_truth_list = None, **kwargs) -> None:
        self.best_indices = indices
        self.config = config
        self.model = copy.deepcopy(model)
        self.data = data
        self.target = target
        
        self.target_class = self.config["attack"]["target_class"]
        self.filtered_output = self.config["attack"]["filtering"]["filtered_output"]
        
        self.initial_weights, _ , _ = read_weights(self.model, self.config)
        self.bounds_model = {key: [get_bounds(torch.squeeze(self.initial_weights[key][index].view(1,-1), dim=0), self.config,numerical=True) for index in indices] for key, indices in self.best_indices.items()}
        self.index_of_interest = 0
        self.ground_truth_list = ground_truth_list
        self.optimisation_objective = self.config["attack"]["objective"]
        self.y_target = self.config["attack"]["objectives"]["kl_div_target"]
        self.target_confidence = self.config["attack"]["target_confidence"]
        
        if "device" in kwargs:
            self.device = "cpu"
        else:
            self.device = next(self.model.parameters()).device
            
    def update_model(self, model):
        parameters = model if isinstance(model, list) else model.parameters()
        
        for old_param, new_param in zip(self.model.parameters(), parameters):
            old_param.data = new_param.data.clone()

    def get_the_model(self, lambda_value):
        for layer, indices in self.best_indices.items():
            for index in indices:
                new_model = modify_the_weights_with_single_neuron(self.model, lambda_value[layer][f"{index}"], self.device, index, layer=layer,pfl=True, server_config=self.config)
        
        self.update_model(new_model)
        
        return self.model
    
    def generate_new_sample(self, samples):
        new_model = copy.deepcopy(self.model)
        for key, weights in samples.items():
            for index, sample in weights.items():
                sample = sample.to(self.device)
                new_model = modify_the_weights_with_single_neuron(copy.deepcopy(new_model), sample, self.device, index, layer=key, pfl=True, server_config=self.config)
            
        logits = new_model(self.data)
        output = F.softmax(logits, dim=1)

        if self.filtered_output:
            idx = 0
            for j in range(len(output)):
                if torch.argmax(output[j]).detach().cpu() == self.target[j] and self.target[j].detach().cpu() in self.config["attack"]["target_class"]:
                    if idx == 0:
                        output_filtered = output[j]
                        logits_filtered = logits[j]
                        output_filtered = output_filtered.reshape(1, output_filtered.shape[0])
                        logits_filtered = logits_filtered.reshape(1, logits_filtered.shape[0])
                    else:
                        output_filtered = torch.cat((output_filtered, output[j].reshape(1,output[j].shape[0])))
                        logits_filtered = torch.cat((logits_filtered, logits[j].reshape(1,logits[j].shape[0])))
                    idx += 1
                    
            if "output_filtered" not in locals():
                output_filtered = output
                logits_filtered = logits

            output_filtered = torch.tensor(output_filtered)
            logits_filtered = torch.tensor(logits_filtered)
            
        else:
            output_filtered = output
            logits_filtered = logits

        if len(output_filtered.shape) < 1:
            output_filtered = output_filtered.reshape(1, output_filtered.shape[0]) 
            logits_filtered = logits_filtered.reshape(1, logits_filtered.shape[0]) 
        
        target = confidence_level(output_filtered, target_confidence=self.target_confidence, classes=output.shape[1])
        
        if self.optimisation_objective == "kl_div":
            score = kl_divergence(output_filtered, target)
            score = 10 if np.isnan(score) else score
            
            if self.config["attack"]["use_logits"]:
                score = logits_score(logits_filtered, self.target)
                
        if self.optimisation_objective == "js_div":            
            score = np.mean(jensenshannon(output_filtered.detach().cpu().numpy(), target.detach().cpu().numpy(), axis=1))

        return score
    
    def generate_initial_data(self):
        if self.optimisation_objective == "kl_div":
            kl_div_scores = []
        elif self.optimisation_objective == "mutual_info":
            mi_scores = []
        elif self.optimisation_objective == "js_div":
            js_div_scores = []
            
        candidates = {}
            
        for i in range(0, self.config["attack"]["initial_dataset"]):
            
            if i==0:
                samples = {key:{f"{index}": torch.squeeze(torch.zeros_like(self.initial_weights[key][index]).view(1,-1), dim=0) for index in indices} for key, indices in self.best_indices.items()}
            else:
                for layer, indices in self.best_indices.items():    
                    neurons = {}
                    for key, index in enumerate(indices):
                        lambda_value = torch.squeeze(torch.zeros_like(samples[layer][f"{index}"]), dim=0)
                        for idx, value in enumerate(lambda_value):
                            high = self.bounds_model[layer][key][1][idx]
                            low = self.bounds_model[layer][key][0][idx]
                            lambda_value[idx] = torch.rand(1) * (high - low) + low
                        neurons[f"{index}"] = lambda_value
                    samples[layer] = neurons
            
            new_model = copy.deepcopy(self.model)
            for key, weights in samples.items():
                for index, sample in weights.items():
                    sample = sample.to(self.device)
                    new_model = modify_the_weights_with_single_neuron(copy.deepcopy(new_model), sample, self.device, index, layer=key, pfl=True, server_config=self.config)    
                
            self.data = self.data.to(next(self.model.parameters()).device)
            
            logits = new_model(self.data)
            output = F.softmax(logits, dim=1)
            
            if self.filtered_output:
                idx = 0
                for j in range(len(output)):
                    if torch.argmax(output[j]).detach().cpu() == self.target[j] and self.target[j].detach().cpu() in self.config["attack"]["target_class"]:
                        if idx == 0:
                            output_filtered = output[j]
                            logits_filtered = logits[j]
                            output_filtered = output_filtered.reshape(1, output_filtered.shape[0])
                            logits_filtered = logits_filtered.reshape(1, logits_filtered.shape[0])
                        else:
                            output_filtered = torch.cat((output_filtered, output[j].reshape(1,output[j].shape[0])))
                            logits_filtered = torch.cat((logits_filtered, logits[j].reshape(1,logits[j].shape[0])))
                        idx += 1
                try:
                    output_filtered = torch.tensor(output_filtered)
                    logits_filtered = torch.tensor(logits_filtered)
                except UnboundLocalError:
                    return False, False
                
                if "output_filtered" not in locals():
                    output_filtered = output
                
            else:
                output_filtered = output
                logits_filtered = logits

            if len(output_filtered.shape) < 1:
                output_filtered = output_filtered.reshape(1, output_filtered.shape[0]) 
                logits_filtered = logits_filtered.reshape(1, logits_filtered.shape[0]) 
            
            target = confidence_level(output_filtered, target_confidence=self.target_confidence, classes=output.shape[1])
            
            target = target.to(next(self.model.parameters()).device)
            
            if self.optimisation_objective == "kl_div":
                kl_div_score = kl_divergence(output_filtered, target)
                kl_div_score = 10 if np.isnan(kl_div_score) else kl_div_score
                
                if self.config["attack"]["use_logits"]:
                    kl_div_score = logits_score(logits_filtered, self.target)
                
                kl_div_scores.append(kl_div_score)
                
            if self.optimisation_objective == "js_div":                
                score = np.mean(jensenshannon(output_filtered.detach().cpu().numpy(), target.detach().cpu().numpy(), axis=1))
            
                js_div_scores.append(score)
                
            
            if i==0:
                for layer, indices in self.best_indices.items():
                    k=0
                    candidates[layer] = {f"{index}" : samples[layer][f"{index}"].reshape(1,samples[layer][f"{index}"].shape[0]) for index in indices}
            else:
                for layer, indices in self.best_indices.items():
                    k=0
                    for index in indices:
                        candidates[f"{layer}"][f"{index}"] =  torch.cat((candidates[f"{layer}"][f"{index}"], samples[layer][f"{index}"].reshape(1,samples[layer][f"{index}"].shape[0])))
                        k+=1
            
        if self.optimisation_objective == "kl_div":
            kl_div_scores = torch.from_numpy(np.array(kl_div_scores))
            kl_div_scores = kl_div_scores.to(device=self.device, dtype=torch.double)
            scores = kl_div_scores
            
        if self.optimisation_objective == "js_div":
            js_div_scores = torch.from_numpy(np.array(js_div_scores))
            js_div_scores = js_div_scores.to(device=self.device, dtype=torch.double)
            scores = js_div_scores
            
        del new_model, output_filtered, samples
            
        return candidates, scores
    
    def obj_callable(self, y, X=None):
        return torch.linalg.norm(y.view(-1,1) - self.y_target, 2, dim=-1)
    
    def model_initialisation(self, x_train, y_train):
        model = SingleTaskGP(x_train.to(dtype=torch.double), y_train.view(-1,1).to(dtype=torch.double))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        model = model.cuda()
        mll = mll.cuda()
        
        return model, mll
    
    @torch.compile(mode="reduce-overhead", disable=True) 
    def optimize_acqui_func_and_get_observation(self, model_list, train_x, best_y):
        MC_SAMPLES = 32
        RAW_SAMPLES = 256
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        
        def obj_callable(xi, X):
            model_gt = self.ground_truth_list[self.layer_of_interest][f"{self.index_of_interest}"].cuda()
            obj = model_gt(X.reshape(1, X.shape[2])).mean
            
            dt = torch.linalg.norm(xi - obj, 1, dim=-1)
            
            return dt
        
        objective = GenericMCObjective(
            objective=obj_callable,
        )
        
        candidates = {}
        
        for layer, indices in self.best_indices.items():
            self.layer_of_interest = layer
            for key, index in enumerate(indices):
                self.index_of_interest = index
                idx = 0
                
                model, mll = model_list[layer][f"{index}"]["model"], model_list[layer][f"{index}"]["mll"]
                fit_gpytorch_model(mll)
                
                acq_func = qExpectedImprovement(
                    model=model,
                    objective=objective,
                    sampler=sampler,
                    best_f = 0.005
                )
                
                candidate, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=self.bounds_model[layer][key].cuda(),
                    q=1,
                    num_restarts=5,
                    raw_samples=RAW_SAMPLES,  
                    options={"batch_limit": 1, "maxiter": 10}
                )
                
                if layer not in candidates:
                    candidates[layer] = {f"{index}" : candidate}
                else:
                    candidates[layer][f"{index}"] = candidate
                
                idx+=1
        
        return candidates

    @torch.compile(mode="reduce-overhead", disable=True) 
    def time_for_magic(self, **kwargs):
        if "runs" in kwargs:
            n_runs = kwargs["runs"]
            
        candidates, scores = self.generate_initial_data()
        best_y = self.obj_callable(scores).min().item()
        
        lambdas = []
        
        i=1
        
        while i < n_runs:
            print(f"\r Optimisation Loop {i:03} | Best y: {best_y :.5f} | Target: {self.y_target:.5f} | Last Score: {scores[-1]:.5f}", end='\r')

            model_list = {}
            for layer, indices in self.best_indices.items():
                for index in indices:
                    model, mll = self.model_initialisation(candidates[layer][f"{index}"], scores)
                    
                    if layer not in model_list:
                        model_list[layer] =  {f"{index}" : {"model":model, "mll":mll}}
                    elif index not in model_list[layer]:
                         model_list[layer][f"{index}"] = {"model":model, "mll":mll}
                    else:
                        model_list[layer] =  {f"{index}" : {"model":model, "mll":mll}}

            new_candidates = self.optimize_acqui_func_and_get_observation(model_list, train_x=candidates, best_y=best_y)
            
            for layer, value in new_candidates.items():
                for index, values in value.items():
                    candidates[layer][index] = torch.cat(( candidates[layer][index], values))

            score = self.generate_new_sample(new_candidates)
            
            scores = torch.cat((scores, torch.tensor([score], device=self.device)))
            
            best_y = self.obj_callable(scores).min().item()
            
            if best_y < (1+0.1)*self.y_target and not (self.y_target == 0):
                best_y += 0.5*self.y_target
                self.y_target = 0.5*self.y_target
            
            i+=1
            
        best_index = (self.obj_callable(scores) == self.obj_callable(scores).min().item()).nonzero(as_tuple=True)[0][0].item()
        best_score = scores[best_index]
        
        lambdas = {}
        for key, values in candidates.items():
            lambdas[key] = {}
            for index, value in values.items():
                lambdas[key][index] = value[best_index]
            
        print(f"Best Score: {best_score :.5f}, Best y: {best_y :.5f}")
        
        return self.get_the_model(lambdas), scores, lambdas, model_list