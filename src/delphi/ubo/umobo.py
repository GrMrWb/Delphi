import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon

from src.delphi.utils import *
from src.delphi.utils.bo_libraries import *

Tensor = torch.Tensor
    
class LayerOptUMOBO:
    def __init__(self, config, indices, model, data, target, ground_truth_list = None, **kwargs):
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
        self.optimisation_objectives = self.config["attack"]["multi_obj_bo"]["objectives"]
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
        scores = {}
        new_model = copy.deepcopy(self.model)
        for key, weights in samples.items():
            for index, sample in weights.items():
                sample = sample.to(self.device)
                new_model = modify_the_weights_with_single_neuron(copy.deepcopy(new_model), sample, self.device, index, layer=key, pfl=True, server_config=self.config)
            
        logits = new_model(self.data)
        output = F.softmax(logits, dim=1)
        accu = 0
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
                
                if self.target[j].detach().cpu() in self.config["attack"]["target_class"]:
                    accu+=1
                    
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
        
        # Generate the target confidence level
        target = confidence_level(output_filtered, target_confidence=self.target_confidence, classes=output.shape[1])
                
        if "kl_div" in self.optimisation_objectives:
            score = kl_divergence(output_filtered, target)
            score = 10 if np.isnan(score) else score
            
            if self.config["attack"]["use_logits"]:
                score = logits_score(logits_filtered, self.target)
            
            scores["kl_div"] = score
            
        if "mutual_info" in self.optimisation_objectives:
            score = torch.Tensor([mi_BO(output_filtered)])
            
            scores["mutual_info"] = score
            
        if "mutual_info" in self.optimisation_objectives:
            score = torch.Tensor([mi_BO(output_filtered)])
            
            scores["mutual_info"] = score
            
        if "wasserstein" in self.optimisation_objectives:
            score = torch.Tensor([wasserstein_matrix(output, target)])
            scores["wasserstein"] = score
        
        if "js_div" in self.optimisation_objectives:            
            score = np.mean(jensenshannon(output_filtered.detach().cpu().numpy(), target.detach().cpu().numpy(), axis=1))
        
            scores["js_div"] = score
            
        if "accuracy" in self.optimisation_objectives:
            score = len(output_filtered) / accu
            scores["accuracy"] = score
            
            
        return scores
    
    def generate_initial_data(self,):
        candidates = {}
        scores = {objective : [] for objective in self.optimisation_objectives}
        
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
            accu = 0
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
                    if self.target[j].detach().cpu() in self.config["attack"]["target_class"]:
                        accu+=1
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
            
            if "kl_div" in self.optimisation_objectives:
                score = kl_divergence(output_filtered, target)
                score = 10 if np.isnan(score) else score
                
                if self.config["attack"]["use_logits"]:
                    score = logits_score(logits_filtered, self.target)
                
                scores["kl_div"].append(score)
                
            if "mutual_info" in self.optimisation_objectives:
                score = torch.Tensor([mi_BO(output_filtered)])
                
                scores["mutual_info"].append(score.numpy())
                
            if "wasserstein" in self.optimisation_objectives:
                score = torch.Tensor([wasserstein_matrix(output, target)])
                
                scores["wasserstein"].append(score)
            
            if "js_div" in self.optimisation_objectives:            
                score = np.mean(jensenshannon(output_filtered.detach().cpu().numpy(), target.detach().cpu().numpy(), axis=1))
            
                scores["js_div"].append(score)
                
            if "accuracy" in self.optimisation_objectives:
                score = len(output_filtered) / accu
                scores["accuracy"].append(score)
                                
            
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
                    
        for obj in scores.keys():
            scores[obj] = torch.from_numpy(np.array(scores[obj]))
            scores[obj] = scores[obj].to(device=self.device, dtype=torch.double)
        
        del new_model, output_filtered, samples
        
        return candidates, scores
        
    def obj_callable(self, y, X=None):
        return torch.linalg.norm(y.view(-1,1) - self.y_target, 2, dim=-1)
    
    def model_iniatilisation(self, x, y_obj_1st, y_obj_2nd):
        y_obj_1st = y_obj_1st.to(dtype=x.dtype)
        y_obj_2nd = y_obj_2nd.to(dtype=x.dtype)
        
        model_obj_1st = SingleTaskGP(x, y_obj_1st.view(-1,1))
        model_obj_2nd = SingleTaskGP(x, y_obj_2nd.view(-1,1))
        
        model = ModelListGP(model_obj_1st, model_obj_2nd)
        
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        model = model.cuda()
        mll = mll.cuda()
        
        return model, mll
    
    def optimize_qehvi_and_get_observation(self, model_list, train_x, ref_point):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
        # partition non-dominated space into disjoint rectangles
        MC_SAMPLES = 128
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        
        ref_point = ref_point.to(self.device)
        
        
        candidates = {}
        
        for layer, indices in self.best_indices.items():
            self.layer_of_interest = layer
            for key, index in enumerate(indices):
                self.index_of_interest = index
                idx = 0
                
                model, mll = model_list[layer][f"{index}"]["model"], model_list[layer][f"{index}"]["mll"]
                fit_gpytorch_model(mll)
        
                with torch.no_grad():
                    pred = model.posterior(train_x[layer][f"{index}"]).mean
                
                partitioning = FastNondominatedPartitioning(
                    ref_point=ref_point,
                    Y=pred,
                )
                
                acq_func = qExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=sampler,
                )
                # optimize
                candidate, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=self.bounds_model[layer][key].cuda(),
                    q=1,
                    num_restarts=2,
                    raw_samples=64,  # used for intialization heuristic
                    options={"batch_limit": 5, "maxiter": 200},
                    sequential=True,
                )
                
                if layer not in candidates:
                    candidates[layer] = {f"{index}" : candidate}
                else:
                    candidates[layer][f"{index}"] = candidate
                
                idx+=1
        
        
        return candidates
    
    def time_for_magic(self, **kwargs) -> Tensor:
        from botorch.utils.multi_objective.pareto import is_non_dominated
        
        print("==== Start of Optimisation ====")
        n_runs= 150
        
        if "runs" in kwargs:
            n_runs = kwargs["runs"]
        
        hvs_list = []
        
        candidates, scores = self.generate_initial_data()
        obj_names = scores.keys()
        
        ref_point = torch.tensor([0.01, 0.5]) if "accuracy" in self.optimisation_objectives else  torch.tensor([0.01, 0.25])
        ref_point = ref_point.to(self.device)
        hv = Hypervolume(ref_point=ref_point)
        
        objs_1st, objs_2nd, lambdas = scores[list(obj_names)[0]].to(self.device), scores[list(obj_names)[1]].to(self.device), candidates
        
        objective = torch.cat((objs_1st.view(-1,1), objs_2nd.reshape(objs_2nd.shape[0]).view(-1,1)), dim=1)
        objective = objective.to(self.device)
        
        feas_train_obj = -1*objective
        pareto_mask = is_non_dominated(feas_train_obj)
        pareto_y = feas_train_obj[pareto_mask]
        # compute hypervolume
        volume = hv.compute(pareto_y)
        hvs_list.append(volume)

        best_y = 0.0001

        for i in range(n_runs):
            torch.cuda.empty_cache()
            
            print(f"\r Optimisation Loop {i:03} | Best y: {best_y :.5f} | Target: {self.y_target:.5f} | Last Score: {volume:.5f}", end='\r')

            model_list = {}
            for layer, indices in self.best_indices.items():
                for index in indices:
                    model, mll = self.model_iniatilisation(candidates[layer][f"{index}"], objs_1st, objs_2nd)
                    
                    if layer not in model_list:
                        model_list[layer] =  {f"{index}" : {"model":model, "mll":mll}}
                    elif index not in model_list[layer]:
                         model_list[layer][f"{index}"] = {"model":model, "mll":mll}
                    else:
                        model_list[layer] =  {f"{index}" : {"model":model, "mll":mll}}
            
            new_candidates = self.optimize_qehvi_and_get_observation(model_list, candidates, ref_point=ref_point)
            
            for layer, value in new_candidates.items():
                for index, values in value.items():
                    candidates[layer][index] = torch.cat(( candidates[layer][index], values))
                
            result = self.generate_new_sample(new_candidates)
            obj_1st, obj_2nd = result[list(obj_names)[0]], result[list(obj_names)[1]]
            
            objs_1st = torch.cat((objs_1st, torch.tensor(obj_1st, device=objs_1st.device).view(1)))
            objs_2nd = torch.cat((objs_2nd, torch.tensor(obj_2nd, device=objs_2nd.device).unsqueeze(0)))
            
            objective=torch.cat((objs_1st.view(-1,1), objs_2nd.view(-1,1)), dim=1)
            objective = objective.to(self.device)
            ref_point = ref_point.to(self.device)
            
            bd = DominatedPartitioning(ref_point=ref_point, Y=objective)
            volume = bd.compute_hypervolume().item()
            
            scores[list(obj_names)[0]] = objs_1st
            scores[list(obj_names)[1]] = objs_2nd
            
            feas_train_obj = -1*objective
            pareto_mask = is_non_dominated(feas_train_obj)
            pareto_y = feas_train_obj[pareto_mask]

            hvs_list.append(volume)
        
        
        best_index = (scores[list(obj_names)[0]] == -1*pareto_y[0][0]).nonzero().item()
        
        lambdas = {}
        for key, values in candidates.items():
            lambdas[key] = {}
            for index, value in values.items():
                lambdas[key][index] = value[best_index]
        
        
        print("===== End of Optimisation =====")
        return self.get_the_model(lambdas), hvs_list, lambdas, model_list