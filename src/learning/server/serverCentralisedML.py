import os, shutil
import copy
import datetime
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.delphi.strategy import *

from src.dataset import source

from src.learning import models as AvailableModels
from src.learning.utils import choose_loss_fn, choose_optimizer

from src.evaluation.uncertainty_quant import evaluate_predictions

class ServerCentralisedML:
    def __init__(self, config, seed=0, checkpoint=False, checkpoint_path="") -> None:
        super().__init__()
        self.server_config = copy.deepcopy(config)
        self.prefix = "{}_".format(self.server_config["configuration"])
        
        # Dataset
        self.selection = self.server_config["collection"]["selection"]
        self.train_batch_size = self.server_config["collection"]["datasets"][self.selection]["training_batch_size"]
        self.test_batch_size = self.server_config["collection"]["datasets"][self.selection]["testing_batch_size"]
        self.val_size = self.server_config["collection"]["datasets"][self.selection]["val_size"]
        
        self.training_dataset = None
        self.validation_dataset = None
        self.testing_dataset = None
        
        # Model choice
        if config[f"{self.prefix}config"]["global_model"]['type'] in AvailableModels.__all__:
            self.model = getattr(AvailableModels, self.server_config[f"{self.prefix}config"]["global_model"]['type'])
        else:
            model = self.server_config[f"{self.prefix}config"]["global_model"]['type']
            raise NotImplementedError(f'This model {model} has not been implemented ')
        
        # Optimisation
        self.optimizer = choose_optimizer(self.model.parameters())# "adam", learning_rate=0.01)
        self.loss_function = choose_loss_fn()
        
        # Early Stopping
        self.early_stopping = self.server_config[f"{self.prefix}config"]["early_stopping"]
        if self.early_stopping:
            self.chances = 0
            self.patience = 50
            self.stop = False
        self.previous_loss = 100
        
        # Checkpoint
        self.path = checkpoint_path
        self.results_path = None
        self.checkpoint = checkpoint
        
        # Settings
        self.epochs = self.server_config[f"{self.prefix}config"]["epochs"]
        self.device = self.server_config["device"]
        self.seed = seed
        
        self.validation_performance = [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric",
                "confidence", "mi", "correctness", "entropy", "variability", "loss"
            ]
        ]
        self.testing_performance = [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric",
                "confidence", "mi", "correctness", "entropy", "variability", "loss"
            ]
        ]
        
        self.analyser = [
            [
                "client_id", "layer_mean_before", "layer_std_before","layer_mean_after", "layer_std_after", "feature_similarity", "layer_similarity"
            ]
        ]
        
        dataset, testing_dataset = getattr(source, f"feed_server_with_data_{self.selection}")(self.val_size, self.train_batch_size, config, CML=True)
        training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [int((1-self.val_size)*len(dataset)), int((self.val_size)*len(dataset))])
        
        self.training_dataset = DataLoader(training_dataset, batch_size=self.train_batch_size , shuffle=True) 
        self.validation_dataset = DataLoader(validation_dataset, batch_size=self.train_batch_size , shuffle=True) 
        self.testing_dataset = DataLoader(testing_dataset, batch_size=self.test_batch_size , shuffle=True)
    
    @torch.compile(mode="reduce-overhead", disable=True)
    def fit(self):
        torch._dynamo.config.suppress_errors = True
        torch.cuda.empty_cache()
        
        old_model = copy.deepcopy(self.model)
        self.model = self.model.to(self.device)
        
        size = int(len(self.training_dataset.dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])
        
        for epoch in range(0,self.epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.training_dataset):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
            
                output = self.model(data)

                loss_func = self.loss_function(output, target)
                
                loss_func.backward()
                self.optimizer.step()        
                
                print('\rEpoch {:02} | Loss: {:.5f} | Batch: {:03}/{:03}'.format(epoch, loss_func.item(), batch_idx, size), end='\r')
                
                del target, data, output
                
                torch.cuda.empty_cache()
                
            layer_distribution_before = layer_weight_distribution(old_model, self.server_config)
            layer_distribution_after = layer_weight_distribution(self.model, self.server_config)
            
            self.analyser.append(
                [
                    layer_distribution_before[0],
                    layer_distribution_before[1],
                    layer_distribution_after[0],
                    layer_distribution_after[1],
                    "no need",
                    layer_cosine_similarity(old_model, self.model, self.server_config),
                ]
            )
            
            self.evaluation(validation=True)
            self.evaluation(testing=True)
            
            old_model = copy.deepcopy(self.model)
            
            if epoch < (self.epochs-1):
                self.save_checkpoint(epoch)
               
            print('\rEpoch {:02} | Loss: {:.5f} | Batch: {:03}/{:03}'.format(epoch, loss_func.item(), batch_idx, size), end='\n')
            
        self.evaluation(testing=True)
        self.log_results()
        
    def training(self, **kwargs):
        pass
        
    def evaluation(self, **kwargs):
        if 'validation' in kwargs:
            dataset = self.validation_dataset
        else:
            dataset = self.testing_dataset
            
        accuracy, precision, recall, aleatoric, confidence = 0, 0, 0, 0, 0
        mi, correctness, entropy, variability, matrix, loss = 0, 0, 0, 0, 0,0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataset):
                data, target = data.to(self.device), target.to(self.device)
                self.model= self.model.to(self.device)
                output = self.model(data)
                loss_func = self.loss_function(output, target)
                
                probabilities = F.softmax(output, dim=1)
                
                results = evaluate_predictions(target, probabilities)
                
                accuracy += results["accuracy"]
                precision += results["precision"]
                recall += results["recall"]
                aleatoric += results["aleatoric"]
                confidence += results["confidence"]
                mi += results["mi"]
                correctness += results["correctness"]
                entropy += results["entropy"]
                variability += results["variability"]
                loss += loss_func.item()
                # matrix += results["matrix"]
                del target, data
                
        accuracy /= batch_idx + 1
        precision /= batch_idx + 1
        recall /= batch_idx+1
        aleatoric /= batch_idx + 1
        confidence /= batch_idx + 1
        mi /= batch_idx + 1
        correctness /= batch_idx + 1
        entropy /= batch_idx + 1
        variability /= batch_idx + 1
        loss /= batch_idx + 1
        # self.matrix = matrix
        
        if 'validation' in kwargs:            
            self.validation_performance.append([
                f'Server',
                accuracy,
                precision,
                recall,
                aleatoric,
                confidence,
                mi,
                correctness,
                entropy,
                variability,
                loss
            ])
            
        else:
            self.testing_performance.append([
                f'Server',
                accuracy,
                precision,
                recall,
                aleatoric,
                confidence,
                mi,
                correctness,
                entropy,
                variability,
                loss
            ])
        
    def save_checkpoint(self, training_round):      
        model_type = self.server_config[f"{self.prefix}config"]["global_model"]['type']
        
        pd.DataFrame(self.analyser).to_csv(f"{self.path}/analyser.csv")
        pd.DataFrame(self.validation_performance).to_csv(f"{self.path}/validation_performance.csv")
        pd.DataFrame(self.testing_performance).to_csv(f"{self.path}/testing_performance.csv")
        
        torch.save(self.model.state_dict(), f"{self.path}/{model_type}_model.pt")
        
        np.savetxt(f"{self.path}/epoch.txt", [training_round+1])
        
        with open(f'{self.path}/config.yaml', 'w+') as outfile:
            yaml.dump(self.server_config, outfile)
        
    def log_results(self):
        dataset = self.server_config["collection"]["selection"]
        d = datetime.datetime.now()
        
        split_path = self.path.split("/")
        path = f"results/runs/{d.month}{d.day}{d.hour}{d.minute}/{split_path[6]}"
        
        if not os.path.isdir(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
        
        model_type = self.server_config[f"{self.prefix}config"]["global_model"]['type']
        
        pd.DataFrame(self.analyser).to_csv(f"{path}/analyser.csv")
        pd.DataFrame(self.validation_performance).to_csv(f"{path}/validation_performance.csv")
        pd.DataFrame(self.testing_performance).to_csv(f"{path}/testing_performance.csv")
        
        torch.save(self.model.state_dict(), f"{path}/{model_type}_model.pt")
        
        self.results_path = path