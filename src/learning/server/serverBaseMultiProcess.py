import copy
import logging
import torch
import torch.nn.functional as F
import numpy as np
import time
import pandas as pd
import datetime

import os, shutil, signal

from torch.distributed.elastic.multiprocessing import Std, start_processes

import torch.multiprocessing as mp

from src.learning import models as AvailableModels
from src.learning.utils import *

from src.evaluation.data_iq import DataIQ_Torch
from src.evaluation.uncertainty_quant import evaluate_predictions
from src.evaluation.utils import EvaluationMetrices, EvaluationConvergence
from src.dataset.data import *

logger = logging.getLogger(__name__)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def trainer(enganging_criteria, client, training_dataset, j):
    if client.trainable and not client.is_adversary:
        if enganging_criteria == None:
            # client.compile_model()
            client.train(training_dataset, epoch=j)
            # client.model = client.model.to('cpu')
            # weights.append(client.model.state_dict())
        else:
            if j < enganging_criteria:
                # client.compile_model()
                client.train(training_dataset, epoch=j)
                # client.model = client.model.to('cpu')
                # weights.append(client.model.state_dict())
            else:
                client.is_client_adversary()
                client.train(training_dataset, epoch=j)
                # client.model = client.model.to('cpu')
                # weights.append(client.model.state_dict())
                
    return copy.deepcopy(client)

def tester(client, dataset):
    client.testing(dataset)
    
    return client

class ServerBaseMultiProcess:
    def __init__(self, config, seed=0, checkpoint=False) -> None:
        super().__init__()
        self.server_config = config
        self.prefix = "{}_".format(self.server_config["configuration"])
        # Workers and clients
        self.num_workers = self.server_config["learning_config"]["K"]
        self.frac_clients = self.server_config["learning_config"]["C"]
        self.workers = self.frac_clients * self.num_workers        
        self.idxs_users = np.random.choice(range(self.num_workers), max(int(self.workers), 1), replace=False)  
        
        # Federaterd Leaning global model configuration
        self.rounds = self.server_config["learning_config"]["rounds"]
        self.aggregation = self.server_config["learning_config"]['aggregator']
        self.enganging_criteria = self.server_config["learning_config"]["enganging_criteria"]
        
        self.available_aggregation = self.server_config["learning_config"]["available_aggregations"]
        
        # Datasets and Clients
        self.list_of_clients = []
        self.datasets = []
        self.testing_dataset = None
        
        self.channels = self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["channels"]
        self.explainable = self.server_config["learning_config"]["global_model"]['explainable']
        
        # Path and Device
        self.path = self.server_config["learning_config"]["global_model"]['path1'] if not self.explainable else self.server_config["learning_config"]["global_model"]['path2']
        self.device = self.server_config["device"]
        
        # Model choice
        if config["learning_config"]["global_model"]['type'] in AvailableModels.__all__:
            try:
                self.model = getattr(AvailableModels, self.server_config["learning_config"]["global_model"]['type'])(channels = self.channels)
            except TypeError:
                self.model = getattr(AvailableModels, self.server_config["learning_config"]["global_model"]['type'])
        else:
            model = self.server_config["learning_config"]["global_model"]['type']
            raise NotImplementedError(f'This model {model} has not been implemented ')
        
        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        self.loss = []
        self.accuracy = []
        
        self.validation_results=[
            [
                "client_id", "accuracy", "precision","recall", "aleatoric",
                "confidence", "mi", "correctness", "entropy", "variability",
            ]
        ]
        
        self.testing_results= [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric",
                "confidence", "mi", "correctness", "entropy", "variability",
            ]
        ]

        # Final Setup
        self.seed = seed
        self.checkpoint = checkpoint
        self.setup_dataset()
        
        if not os.path.isdir(f"{os.getcwd()}/logs"):
            os.mkdir(f"{os.getcwd()}/logs/")
        
    def setup_dataset(self):
        # Creating the Dataset
        datasource = DataSources(self.server_config, True)
        self.datasets = datasource.get_training_data()
        self.validation_datasets = datasource.get_validation_data()
        self.testing_dataset = datasource.testset
        
        self.unique_labels = datasource.unique_labels
    
    def send_parameters(self):
        for client in self.list_of_clients:
            client.set_parameters(self.model)
    
    def extract_processes(self, results):
        for result in results:
            self.list_of_clients[results[result].client_id - 1] = copy.deepcopy(results[result])
    
    def multithread_training(self,j, args, workers=None):
        
        # trainer(self.enganging_criteria, self.list_of_clients[1], self.datasets[1], j)
        direc = f"{os.getcwd()}/logs/{self.prefix}{self.server_config[f'{self.prefix}config']['global_model']['type']}train{j}"
        shutil.rmtree(direc, ignore_errors=True)
        os.mkdir(direc)
        
        ctx = start_processes(
                name="trainer",
                entrypoint=trainer,
                args=args,
                envs={i : {} for i in range(0, self.workers if workers is None else workers)},
                log_dir=direc,
                redirects=Std.ERR, # write all worker stdout/stderr to a log file
                tee=Std.NONE
            )
        
        # results = ctx.wait(timeout=-15*10000).return_values

        # # waits for all copies of trainer to finish
        results = ctx.wait().return_values
        try:
            pass
        except:
            try:
                results = ctx.wait(timeout=-15*60).return_values
            except:
                logger.error('WE ARE FUCKED')
        
        self.extract_processes(results)
        
        ctx.close()
        pids = ctx.pids()
        print(pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                pass
        del ctx, results
        
    def multithread_testing(self, args, workers=None, **kwargs):
        
        j = kwargs["round"] if "round" in kwargs else 0
        
        direc = f"{os.getcwd()}/logs/{self.prefix}{self.server_config[f'{self.prefix}config']['global_model']['type']}test{j}"
        shutil.rmtree(direc, ignore_errors=True)
        os.mkdir(direc)
        
        ctx = start_processes(
                name="tester",
                entrypoint=tester,
                start_method='spawn',
                args=args,
                envs={i : {"Testing Model":i} for i in range(0, self.workers if workers is None else workers)},
                log_dir=direc,
                redirects=Std.ERR, # write all worker stdout/stderr to a log file
                tee=Std.NONE
            )

        # # waits for all copies of trainer to finish
        results = ctx.wait().return_values
        try:
            pass
        except:
            try:
                results = ctx.wait(timeout=-15*60).return_values
            except:
                logger.error('WE ARE FUCKED')
        
        self.extract_processes(results)
        
        ctx.close()
        pids = ctx.pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                pass
        # print(pids)
        # for pid in pids:
        #     os.kill(pids[pid], signal.SIGTERM)
        
        del ctx, results
        
    
    def fit(self) -> None:
        
        if self.checkpoint:
            training_round = int(np.loadtxt(f"{os.getcwd()}/checkpoint/{self.server_config['name']}/epoch.txt"))
        else:
            training_round = 0
            
        for j in range(0,self.rounds):
            print('======= Start of Round {} ======='.format(j+1))
            logger.info('==================\nStart of Round {}\n================\n'.format(j+1))
            weights = [] if self.aggregation in self.available_aggregation else {}
            
            # Normal Clients
            args = {i : (copy.deepcopy(self.enganging_criteria), copy.deepcopy(self.list_of_clients[i]), copy.deepcopy(self.datasets[i]), copy.deepcopy(j), ) for i in range(0, self.workers) }
            
            self.multithread_training(j, args)
            
            # Adversarial clients
            if self.enganging_criteria < j:
                args = {}
                counter=0
                for i in range(0, self.workers):
                    if self.list_of_clients[i].is_adversary:
                        args[counter] = (copy.deepcopy(self.enganging_criteria), copy.deepcopy(self.list_of_clients[i]), copy.deepcopy(self.datasets[i]), copy.deepcopy(j))
                        counter +=1
                        
                self.multithread_training(j, args, workers=counter)
            
            # Aggregation
            weights = [self.list_of_clients[i].model.state_dict() for i in range(0, self.workers)]
            self.global_weight = self.aggregator(weights)
            self.model.load_state_dict(self.global_weight)                

            print('======== End of Round {} ========'.format(j+1))
                
            logger.info('GPU Temperature {}'.format(get_gpu_temperature()))
            
            logger.info('==================\Model Evaluation\n================\n')
            
            # Model Evaluation
            # data_iq_testing.on_epoch_end_batch(self.testing_dataset['dataloader'], self.model, device=self.device, attack = self.server_config['evaluation_attack'])
            
            self.global_evaluation(round=j)
            self.personalised_evaluation(round=j)
            print("\n")
            
            # Checkpoint
            if j < (self.rounds-1):
                self.save_checkpoint(j)
            
            try:
                commit_results(self.workers, f"{os.getcwd()}/checkpoint/{self.server_config['name']}")
            except:
                logger.error('Error with checkpoint')
            
            try:
                commit_to_git()
            except:
                logger.error('Error with the Git')

            self.send_parameters()            
        
        self.log_results()

        try:
            commit_to_git()
        except:
            logger.error('Error with the Git')
            
        dataset = self.server_config["collection"]["selection"]
            
        torch.save(self.model.state_dict(), f"{self.server_config['name']}_Server_{dataset}_{self.seed}.pt")
                       
    def aggregator(self, weights) -> dict:
        if self.aggregation == "FedAvg":
            return FedAvg(weights)
        elif self.aggregation == "Krum":
            if not isinstance(weights, dict):
                dict_weights = {idx : weight for idx, weight in enumerate(weights)}
            return Krum(dict_weights, self.workers)
        elif self.aggregation == "FedDyn": 
            self.update_server_state()
            self.aggregate_parameters()
        elif self.aggregation == "Ditto": 
            pass
        elif self.aggregation == "pFedME": 
            return PFedAvg(self.model, self.list_of_clients)
        elif self.aggregation == "PerFedAvg": 
            return FedAvg(weights)
        elif self.aggregation == "FedProx": 
            return FedAvg(weights)
        
    def evaluation(self, **kwargs):
        if 'validation' in kwargs:
            dataset = kwargs['validation']
            dataset = dataset['dataloader']
        else:
            dataset = kwargs['testing']
        
        accuracy, precision, recall, aleatoric, confidence = 0, 0, 0, 0, 0
        mi, correctness, entropy, variability = 0, 0, 0, 0
        
        self.model.eval()        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                data, target = data.to(self.device), target.to(self.device)
                self.model= self.model.to(self.device)
                output = self.model(data)
                
                probabilities = F.softmax(output, dim=1)
                
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
        
        if 'validation' in kwargs:
            self.testing_results.append([
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
            ])
            
        else:
            self.validation_results.append([
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
            ])

        # TODO change the way that I am gathering data  
        
    def global_evaluation(self, **kwargs):
        self.evaluation(testing = self.testing_dataset["dataloader"])
        
        # args = {i : (copy.deepcopy(self.list_of_clients[i]), copy.deepcopy(self.testing_dataset["dataloader"])) for i in range(0, self.workers) }
        
        # self.multithread_testing(args, round = kwargs["round"])
        
        for i in range(0, self.num_workers):
            #TODO check the models for the personlaised federated leanring
            
            self.list_of_clients[i].testing(self.testing_dataset["dataloader"])
        
    def personalised_evaluation(self, **kwargs):
        args = {i : (copy.deepcopy(self.list_of_clients[i]), copy.deepcopy(self.testing_dataset["dataloader"])) for i in range(0, self.workers) }
        
        # self.multithread_testing(args, round = kwargs["round"])
        
        for i in range(0, self.workers):
            #TODO check the models for the personlaised federated leanring

            self.evaluation(validation = self.validation_datasets[i])
            
            self.list_of_clients[i].validation(self.validation_datasets[i])

    def log_results(self):
        dataset = self.server_config["collection"]["selection"]
        d = datetime.datetime.now()
        
        for client in self.list_of_clients:
            pd.DataFrame(client.training_performance).to_csv(f"{self.server_config['name']}_Client_{client.client_id}_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_training_performance")
            pd.DataFrame(client.validation_performance).to_csv(f"{self.server_config['name']}_Client_{client.client_id}_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_validation_performance")
            pd.DataFrame(client.testing_performance).to_csv(f"{self.server_config['name']}_Client_{client.client_id}_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_testing_performance")
            if client.is_adversary:
                pd.DataFrame(client.scores).to_csv(f"{self.server_config['name']}_Client_{client.client_id}_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_LO_scores")
                pd.DataFrame(client.hvs_list).to_csv(f"{self.server_config['name']}_Client_{client.client_id}_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_hvs_list")
        
        pd.DataFrame(self.validation_results).to_csv(f"{self.server_config['name']}_Server_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_validation_performance")
        pd.DataFrame(self.testing_results).to_csv(f"{self.server_config['name']}_Server_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_testing_performance")
        
        # pd.DataFrame(self.unique_labels).to_csv(f"{self.server_config['name']}_Clients_{dataset}_{d.month}{d.day}{d.hour}{d.minute}_unique_labels")
    
    def save_checkpoint(self, training_round):
        path = f"{os.getcwd()}/checkpoint/{self.server_config['name']}"
        
        for client in self.list_of_clients:
            pd.DataFrame(client.training_performance).to_csv(f"{path}/Client_{client.client_id}_training_performance.csv")
            pd.DataFrame(client.validation_performance).to_csv(f"{path}/Client_{client.client_id}_validation_performance.csv")
            pd.DataFrame(client.testing_performance).to_csv(f"{path}/Client_{client.client_id}_testing_performance.csv")
            if client.is_adversary:
                pd.DataFrame(client.scores).to_csv(f"{path}/Client_{client.client_id}_LO_scores.csv")
                pd.DataFrame(client.hvs_list).to_csv(f"{path}/Client_{client.client_id}_hvs_list.csv")
        
        pd.DataFrame(self.validation_results).to_csv(f"{path}/Server_validation_performance.csv")
        pd.DataFrame(self.testing_results).to_csv(f"{path}/Server_testing_performance.csv")
        
        torch.save(self.model.state_dict(), f"{path}/Server_model.pt")
        
        np.savetxt(f"{path}/epoch.txt", [training_round+1])
    

    def validate_clients_personalised_models(self):
        results = []
        accuracy = 0
        loss = 0
        
        for i in range(0, self.workers):
            # try:
            testing_datest = self.validation_datasets[i]
            client = self.list_of_clients[i]
            
            result_normal = client.evaluate_accuracy_loss_model(testing_datest, adversarial=False)
            result_adversarial = client.evaluate_accuracy_loss_model(testing_datest, adversarial=True)
            
            results.append([result_normal, result_adversarial])
            accuracy += result_normal["accuracy"]
            loss += result_normal["loss"]
            # except:
            #     pass
            
        accuracy /= self.workers
        loss /= self.workers
        
        logger.info(f"Validation results for the clients \n{results}\n Accuracy: {accuracy}\n Loss: {loss}")
        
        print(f'\rValidation | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}', end='\n')
        
        self.personalised_accuracy.append(accuracy)
        self.personalised_loss.append(loss)
        
    def evaluate_clients_personalised_models(self):
        results = []
        accuracy, loss = 0, 0
        for i in range(0, self.workers):
            try:
                testing_datest = self.testing_dataset
                client = self.list_of_clients[i]
                
                result_normal = client.evaluate_accuracy_loss_model(testing_datest, adversarial=False, model = "personalised")
                result_adversarial = client.evaluate_accuracy_loss_model(testing_datest, adversarial=True, model = "personalised")
                
                results.append([result_normal, result_adversarial])
                accuracy += result_normal["accuracy"]
                loss += result_normal["loss"]
            except:
                pass
            
        accuracy /= self.workers
        loss /= self.workers
        
        logger.info(f"Testing results for the clients \n{results}\n Accuracy: {accuracy}\n Loss: {loss}")
        
        print(f'\rTesting | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}', end='\n')
        
        self.generalised_accuracy.append(accuracy)
        self.generalised_loss.append(loss)
        