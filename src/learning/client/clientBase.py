import copy
import logging
from typing import Tuple, Union
from collections import OrderedDict
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import time

import config
from src.learning import models as AvailableModels
from src.learning.utils import (
    choose_loss_fn, choose_optimizer,
    PTFedProxLoss, # FedProx
)

from src.evaluation import (
    evaluate_predictions,
    # run_uncertainty_evaluator
)

from src.delphi.strategy import *

logger = logging.getLogger(__name__)

class ClientBase:
    def __init__(self, client_id, device, **kwargs) -> None:
        self.client_id = client_id + 1
        
        # configuration for the client
        self.client_config = copy.deepcopy(config.client_configuration["clients"][self.client_id])
        self.server_config = copy.deepcopy(config.server_configuration)
        self.configuration = self.server_config["configuration"]
        self.experiment = self.server_config["experiment"]
        self.device = self.server_config["device"]
        
        self.early_stopping = self.client_config['early_stopping']
        self.chances = 10
        
        # Check if it is adversary
        self.trainable = True
        self.is_adversary = kwargs["adversary"]
        self.explainable = self.server_config["learning_config"]["global_model"]['explainable']
        
        # Number of Epochs
        self.epochs = self.server_config["learning_config"]["epochs"]

        self.channels = self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["channels"]
        self.unique_labels = None
        
        if self.server_config["learning_config"]["global_model"]["type"] in AvailableModels.__all__:
            try:
                self.model = copy.deepcopy(getattr(AvailableModels, self.server_config["learning_config"]["global_model"]['type'])(channels = self.channels))
            except TypeError:
                self.model = copy.deepcopy(getattr(AvailableModels, self.server_config["learning_config"]["global_model"]['type']))
        else:
            model = self.server_config["learning_config"]["global_model"]["type"]
            raise NotImplementedError(f'This model {model} has not been implemented ')
        
        self.loss = choose_loss_fn(self.client_config["loss"])
        self.optimizer = choose_optimizer(self.model.parameters(), self.client_config["optimizer"])
        
        self.training_performance = []
        self.validation_performance = [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric", "confidence", 
                "mi", "correctness", "entropy", "variability", "loss","ece", "mce"
            ]
        ]
        
        if not self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["iid"]:
            self.validation_performance_niid = [
                [
                    "client_id", "accuracy", "precision","recall", "aleatoric", "confidence", 
                    "mi", "correctness", "entropy", "variability", "loss", "ece", "mce"
                ]
            ]
        
        self.testing_performance = [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric", "confidence", 
                "mi", "correctness", "entropy", "variability", "loss", "ece", "mce"
            ]
        ]
        
        self.best_indices = [] if self.server_config["attack"]["best_features"] == "continues" else [self.get_best_indices()]
        self.attack_engangement = []
        self.gradients = []
        self.distributions_mean = []
        self.distributions_std = []
        self.cka = []
        self.best_indices_arr = []
        
        if self.is_adversary:
            self.scores = []
            self.lambdas = []
            self.hvs_list = []
            self.historical_data = {
                "uncertainty" : [], 
                "accuracy" : [], 
                "variability" : [], 
                "weights" : []
            }
            
        if self.server_config["learning_config"]["uncertainty_evaluation"]:
            self.uncertainty_evaluation_perf = [
                ["mc_dropout", "bnn", "temperature_scaling","probabilistic"]
            ]
        
    
    def compile_model(self)-> None:
        """ It is been only used only with the 

        Raises:
            NotImplementedError: if the model is not correctly in the configuration file or is not implemented, then the model wont be found
        """
        if self.server_config["learning_config"]["global_model"] in AvailableModels:
            self.model = getattr(AvailableModels, self.server_config["learning_config"]["global_model"])(channels = self.channels)
        else:
            raise NotImplementedError(f'This model {self.server_config["learning_config"]["global_model"]} has not been implemented ')
        
        self.optimizer = choose_optimizer(self.model.parameters(),self.client_config["optimizer"])
        
    def set_parameters(self, model, **kwargs):
        if 'default' in kwargs:
            if kwargs["default"]:
                if self.server_config["learning_config"]["global_model"] in AvailableModels:
                    self.model = getattr(AvailableModels, self.server_config["learning_config"]["global_model"])(channels = self.channels)
                else:
                    raise NotImplementedError(f'This model {self.server_config["learning_config"]["global_model"]} has not been implemented ')
                
                self.optimizer = choose_optimizer(self.model.parameters(),self.client_config["optimizer"])
        
        for old_param, new_param in zip(self.model.parameters(),  model.parameters()):
            old_param.data = new_param.data.clone()

    def get_parameters(self, **kwargs):
        if 'personalised' in kwargs:
            if kwargs['personalised']:
                return self.personalised_model.state_dict()
            if kwargs['global']:
                return self.global_model.model.state_dict()
        else:
            return self.model.state_dict()

    def setup_dataset(self, unique_labels):
        self.unique_labels = unique_labels
        if self.is_adversary:
            
            self.server_config["attack"]["target_class"] = self.unique_labels
            
    def train(self,dataset : object, **kwargs) -> list:
        if self.server_config["attack"]["engagement_criteria"]["epoch"] < kwargs["epoch"] and self.is_adversary:
            return self.__client_adversary__(dataset, kwargs["epoch"])
        else:
            return self.__client_normal__(dataset, **kwargs)
    
    def __client_adversary__(self,dataset: object, tr_round:int) -> list:
        return self.__experiment_LO__(dataset, tr_round)
    
    def continue_training(self, dataset, epochs, **kwargs):
        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"]) + 1

        self.model.train()
        start = 0 if epochs==1 else 1
        end = epochs
        for epoch in range(start,end):
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                data, target = data.to(self.device), target.to(self.device)
                self.model = self.model.to(self.device)
                
                loss_func = self.__train__(data, target,batch_idx, size, epoch) if "global_train" in kwargs or not bool(kwargs) else self.__personalised_train__(data, target,batch_idx, size, epoch)
            
            print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Epoch: {:02} '.format(self.client_id, loss_func.item(), batch_idx+1, size, epoch+1), end='\r')
            
        try:
            print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Epoch: {:02} '.format(self.client_id, loss_func.item(), batch_idx+1, size, epoch+1), end='\n')
        except:
            pass
        self.model = self.model.to("cpu")
        return loss_func
    
    def __experiment_LO__(self, dataset: object, tr_round:int) -> list:
        previous_loss = 100
        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"]) + 1

        torch.cuda.empty_cache()
        
        self.model= self.model.to(self.device)
        old_model = copy.deepcopy(self.model)
        
        random_sample = np.random.random_integers(0, len(dataset["dataloader"]))
        
        for batch_idx, (data, target) in enumerate(dataset['dataloader']):
            if batch_idx == random_sample:
                data, target, self.model = data.to(self.device), target.to(self.device), self.model.to(self.device)
                break
    
        loss_func = self.__test__(data, target)
        
        self.model.eval()
        if loss_func < self.server_config["attack"]["engagement_criteria"]["loss"]:
                    
            if self.server_config["attack"]["type_of_optimisation"] == "bayesian_optimisation":
                            
                model, analysis = apply_usobo(
                    self.server_config, 
                    copy.deepcopy(self.model), 
                    dataset['dataloader'], 
                    self.device, 
                    self.historical_data,
                    best_indices=self.best_indices
                ) if self.server_config["attack"]["type"]=="single" else apply_umobo(
                    self.server_config, 
                    copy.deepcopy(self.model), 
                    dataset['dataloader'], 
                    self.device, 
                    self.historical_data, 
                    best_indices=self.best_indices
                )
                
                if isinstance(model, bool):
                    loss_func = self.continue_training(self, dataset, self.epochs)
                else:
                    self.model = copy.deepcopy(model)

                    self.lambdas.append(analysis["lambda_sample"])
                    self.scores.append(analysis["score"].detach().cpu().numpy() if not isinstance(analysis["score"], list) else analysis["score"])
                    self.best_indices.append(analysis["best_indice"])
                    
                    self.attack_engangement.append(tr_round)
                    
            elif self.server_config["attack"]["type_of_optimisation"] == "least_squares":
                model, analysis = apply_les(
                    self.server_config, 
                    copy.deepcopy(self.model), 
                    dataset["dataloader"], 
                    self.device, 
                    kl_div=True, 
                    best_indices=self.best_indices
                )
                
                if isinstance(model, bool):
                    loss_func = self.continue_training(self, dataset, self.epochs)
                else:
                    self.model = copy.deepcopy(model)
                    
                    self.lambdas.append(analysis["lambda_sample"])
                    self.scores.append(analysis["score"])
                    
                    self.attack_engangement.append(tr_round)
                   
            elif self.server_config["attack"]["type_of_optimisation"] == "rl":
                model, analysis = apply_rl(
                    self.server_config, 
                    copy.deepcopy(self.model), 
                    dataset["dataloader"], 
                    self.device,
                    best_indices=self.best_indices
                )
                
                self.model = copy.deepcopy(model)
                self.scores.append(analysis["score"])
                self.attack_engangement.append(tr_round)
                    
            running_loss = self.__test__(data, target)
            
            self.best_indces = analysis["best_indice"]
            
            try:
                print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} '.format(self.client_id, running_loss, batch_idx, size), end='\n')
            except:
                pass
            
        else:
            if self.__class__.__name__ != "ClientDitto":
                loss_func = self.continue_training(dataset, self.epochs)
            else:
                
                self.model = self.model.to(self.device)
                self.personalised_model = self.personalised_model.to(self.device)
                
                loss_func = self.continue_training(dataset, self.plocal_steps, personalised=True)
                
                self.personalised_model = self.personalised_model.to('cpu')
                
                print("\r                                                                                                                              ", end='\r')
                
                loss_func = self.continue_training(dataset, self.epochs, global_train=True)
                
                self.training_performance.append(loss_func.item())
                self.model = self.model.to('cpu')

        print("\rCalculating gradients", end="\r")
        self.gradients.append(get_historical_gradients(self.model, dataset["dataloader"], self.server_config))
        self.gradients.append(get_historical_gradients(old_model, dataset["dataloader"], self.server_config))
        
        print("\rCalculating Distribution", end="\r")
        # mean, std = get_historical_distribution(self.model, self.server_config)
        weights, _, _ = read_weights(self.model, self.server_config)
        
        self.distributions_mean.append(weights)
        
        print("\rCalculating CKA", end="\r")
        # self.cka.append(calculate_cka_model(self.model, old_model,dataset['dataloader'], self.device))
        
        del dataset, old_model
        torch.cuda.empty_cache()
    
    @torch.compile(mode="reduce-overhead", disable=True) 
    def __client_normal__(self, dataset: object,  **kwargs) -> list:
        torch._dynamo.config.suppress_errors = True
        self.model.train()
        
        torch.cuda.empty_cache()
        
        old_model = copy.deepcopy(self.model)
        self.model = self.model.to(self.device)
        
        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])
        
        for epoch in range(0,self.epochs):
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                
                loss_func = self.__train__(data, target,batch_idx, size, epoch)
            
            print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Local Epoch: {:02} '.format(self.client_id, loss_func.item(), batch_idx+1, size, epoch+1), end='\r')
             
        print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Local Epoch: {:02} '.format(self.client_id, loss_func.item(), batch_idx+1, size, epoch+1), end='\n')
        
        self.training_performance.append(loss_func.item())
        
        self.model = self.model.to('cpu')
        
        # print("\rCalculating gradients", end="\r")
        # self.gradients.append(get_historical_gradients(self.model, dataset["dataloader"], self.server_config))
        # self.gradients.append(get_historical_gradients(old_model, dataset["dataloader"], self.server_config))
        
        print("\rCalculating Distribution", end="\r")
        # mean, std = get_historical_distribution(self.model, self.server_config)
        weights, _, _ = read_weights(self.model, self.server_config)
        
        self.distributions_mean.append(weights)
            
        del old_model
        del dataset, loss_func
        
        torch.cuda.empty_cache()
    
    @torch.compile(mode="reduce-overhead", disable=True) 
    def __train__(self, data: object, target: object, batch_idx: int, size: int, epoch:int):           
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
    
        output = self.model(data)

        loss_func = self.loss(output, target)
        
        loss_func.backward()
        self.optimizer.step()        
        
        print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03}'.format(self.client_id,loss_func.item(),batch_idx, size), end='\r')
        
        del target, data, output
        
        torch.cuda.empty_cache()
        
        return loss_func
    
    def __test__(self, data: object, target: object) -> float:
        self.model.eval()
        with torch.no_grad():
            data, target = data.to(self.device), target.to(self.device)
            self.model= self.model.to(self.device)
            
            output = self.model(data)

            loss_func = self.loss(output, target)
            
            del target, data, output
            self.model.to('cpu')
            torch.cuda.empty_cache()
            
        return loss_func.item()
    
    @torch.compile(mode="reduce-overhead", disable=True) 
    def validation(self, dataset, niid=False) -> None:
        accuracy, precision, recall, aleatoric, confidence = 0, 0, 0, 0, 0
        mi, correctness, entropy, variability,loss = 0, 0, 0, 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    self.model= self.model.to(self.device)
                    output = self.model(data)
                    
                    loss_func = self.loss(output, target)
                    
                    probabilities = F.softmax(output, dim=1)
                    loss += loss_func.item()
                    
                    if batch_idx==0:
                        prob_array = probabilities
                        target_array = target
                    else:
                        prob_array = torch.cat([prob_array, probabilities]) 
                        target_array = torch.cat([target_array, target]) 
                except:
                    pass
                    
            results = evaluate_predictions(target_array, prob_array)
                
        accuracy = results["accuracy"]
        precision = results["precision"]
        recall = results["recall"]
        aleatoric = results["aleatoric"]
        confidence = results["confidence"]
        mi = results["mi"]
        correctness = results["correctness"]
        entropy = results["entropy"]
        variability = results["variability"]
        
        loss /= (batch_idx + 1)
        
        result = [
                f'Client_id: {self.client_id}',
                accuracy,
                precision,
                recall,
                aleatoric,
                confidence,
                mi,
                correctness,
                entropy,
                variability,
                loss,
                results["ece"],
                results["mce"],
            ]
        
        if not niid:
            self.validation_performance.append(result)
                    
        else:
            self.validation_performance_niid.append(result)
            
        if self.is_adversary:
            self.historical_data["accuracy"].append(accuracy)
            self.historical_data["variability"].append(variability)
            uncertainty, weights = get_historical_data(self.model,dataset['dataloader'], self.server_config)
            self.historical_data["uncertainty"].append(uncertainty)
            
            if self.best_indices == []:
                indices = self.get_best_indices()
                best_weights = {key : [weights[key][index].flatten().detach().cpu().numpy() for index in value] for key, value in indices.items()}
            else:
                indices = self.best_indices if not isinstance(self.best_indices, list) else self.best_indices[-1]
                    
                best_weights = {key : [weights[key][index].flatten().detach().cpu().numpy() for index in value] for key, value in indices.items()}
            
            if len(self.historical_data["weights"]) == 0:
                self.historical_data["weights"] = [best_weights]
            else:
                self.historical_data["weights"].append(best_weights)
        
        else:
            sample_data, sample_target = [], []
            self.model = self.model.to(self.device)
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                if len(data)<2:
                    break
                data, target = data.to(self.device), target.to(self.device)
                output = F.softmax(self.model(data), dim=1)
                for i in range(len(output)):
                    if torch.argmax(output[i]) == target[i]:
                        sample_data.append(data[i].detach().cpu().numpy())
                        sample_target.append(int(target[i].detach().cpu().float()))
                    
                    if len(sample_target) > 128:
                        break
            
            sample_data, sample_target = torch.tensor(sample_data), torch.tensor(sample_target)
            indices = find_the_best_features_FL(self.model, sample_data, sample_target, server_config=self.server_config)
    
            self.best_indices = indices
                
        self.best_indices_arr.append(self.best_indices)
        
        print(f"\rClient {self.client_id:02} | Validation | Accuracy: {accuracy:.5f} | MI:  {mi:.5f} | Correctness: {correctness:.5f}", end="\r")
               
    @torch.compile(mode="reduce-overhead", disable=True) 
    def testing(self, dataset, niid=False) -> None:
        accuracy, precision, recall, aleatoric, confidence = 0, 0, 0, 0, 0
        mi, correctness, entropy, variability,loss = 0, 0, 0, 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataset):
                data, target = data.to(self.device), target.to(self.device)
                self.model= self.model.to(self.device)
                output = self.model(data)
                
                loss_func = self.loss(output, target)
                
                probabilities = F.softmax(output, dim=1)
                loss += loss_func.item()
                
                if batch_idx==0:
                    prob_array = probabilities
                    target_array = target
                else:
                    prob_array = torch.cat([prob_array, probabilities]) 
                    target_array = torch.cat([target_array, target]) 
                    
            
            results = evaluate_predictions(target_array, prob_array)
                
        accuracy = results["accuracy"]
        precision = results["precision"]
        recall = results["recall"]
        aleatoric = results["aleatoric"]
        confidence = results["confidence"]
        mi = results["mi"]
        correctness = results["correctness"]
        entropy = results["entropy"]
        variability = results["variability"]
        
        loss /= (batch_idx + 1)
        
        result = [
            f'Client_id: {self.client_id}',
            accuracy,
            precision,
            recall,
            aleatoric,
            confidence,
            mi,
            correctness,
            entropy,
            variability,
            loss,
            results["ece"],
            results["mce"],
        ]
        
        self.testing_performance.append(result)
        
        
        print(f"\rClient {self.client_id:02} | Testing | Accuracy: {accuracy:.5f} | MI:  {mi:.5f} | Correctness: {correctness:.5f}", end="\r")            

    # def uncertainty_evaluation(self, training_dataset, validation_dataset, testing_dataset):
    #     scores = run_uncertainty_evaluator(
    #         model=self.model,
    #         train_dataset=training_dataset,
    #         validation_dataet=validation_dataset,
    #         testing_dataset=testing_dataset,
    #         task="classification"
    #     )
        
    #     self.uncertainty_evaluation_perf.append(scores)
        
    def get_best_indices(self):
        weights, _, _ =  read_weights(self.model, self.server_config)
        
        all_indices = { key: [np.random.randint(0, len(weights[key])) for _ in range(0, self.server_config["attack"]["num_of_neurons"])] for key in weights.keys()}
        
        for key, indices in all_indices.items():
            while len(np.unique(indices)) != self.server_config["attack"]["num_of_neurons"]:
                indices = [ np.random.randint(0, len(weights[key])) for _ in range(0, self.server_config["attack"]["num_of_neurons"])]
        
        return all_indices