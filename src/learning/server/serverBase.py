import copy, os, shutil
import logging
import torch
import torch.nn.functional as F
import numpy as np
import time, yaml, json
import pandas as pd
import datetime
import pdb
import networkx as nx

from src.learning import models as AvailableModels
from src.learning.utils import *

from src.evaluation.graph_analysis import FLGraphAnalysis
from src.evaluation.uncertainty_quant import evaluate_predictions
from src.dataset.data import *

from src.delphi.strategy import read_weights

logger = logging.getLogger(__name__)

class ServerBase:
    def __init__(self, config, seed=0, checkpoint=False, checkpoint_path = "") -> None:
        super().__init__()
        self.server_config = copy.deepcopy(config)
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
        self.number_of_adversaries = self.server_config["learning_config"]["number_of_adversaries"]

        self.adversarial_clients = np.random.choice([i for i in range(0, self.num_workers)], self.number_of_adversaries, replace=False)
        logger.debug(f"Number of adversarial clients: {self.adversarial_clients}")
        print(f"Number of adversarial clients: {self.adversarial_clients}")
        
        # Datasets and Clients
        self.list_of_clients = []
        self.training_datasets = []
        self.testing_dataset = None
        self.channels = self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["channels"]
        self.explainable = self.server_config["learning_config"]["global_model"]['explainable']
        self.unique_labels = {}
        
        # Path, Device, Seed
        self.device = self.server_config["device"] # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # # Early Stopping tolerance
        self.early_stopping = self.server_config["learning_config"]["early_stopping"]
        if self.early_stopping:
            self.chances = 0
            self.patience = 50
        self.stop = False
        self.previous_loss = 100
        
        # Model choice
        model_type = config["learning_config"]["global_model"]['type']
        available_models_upper = {m.upper(): m for m in AvailableModels.__all__}

        if model_type.upper() in available_models_upper:
            original_model_name = available_models_upper[model_type.upper()]
            self.model = copy.deepcopy(getattr(AvailableModels, original_model_name))
        else:
            raise NotImplementedError(f'This model {model_type} has not been implemented')
        
        self.loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        self.loss = []
        self.accuracy = []
        
        # Results
        self.validation_results=[
            [
                "client_id", "accuracy", "precision","recall", "aleatoric", "confidence", 
                "mi", "correctness", "entropy", "variability", "loss", "ece", "mce"
            ]
        ]
        
        self.testing_results= [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric", "confidence", 
                "mi", "correctness", "entropy", "variability", "loss", "ece", "mce"
            ]
        ]
        
        self.graph_analysis_results, self.graph_analysis_latex, self.graph_analysis_json = [], [], []
        
        self.best_indices_arr = []
        self.distributions_mean=[]
        
        #checkpoint
        self.path = checkpoint_path
        self.checkpoint = checkpoint
        self.results_path = None 
        
        if self.server_config["learning_config"]["uncertainty_evaluation"]:
            self.uncertainty_evaluation_perf = [
               [ "mc_dropout", "bnn", "temperature_scaling","probabilistic"]
            ]
        
    def setup_dataset(self):
        # Creating the Dataset
        datasource = DataSources(self.server_config, True, adversarial_clients=self.adversarial_clients)
        self.training_datasets = datasource.get_training_data()
        if hasattr(datasource, "users_validation_data_iid"):
            self.validation_datasets_iid = datasource.users_validation_data_iid
            self.validation_datasets_niid = datasource.users_validation_data_niid
            
            counter=0
            for training_data, validation_data_iid, validation_data_niid in zip(self.training_datasets, self.validation_datasets_iid, self.validation_datasets_niid):
                res = []
                [res.append(x) for x in training_data["unique_labels"] if x not in res]
                [res.append(x) for x in validation_data_niid["unique_labels"] if x not in res]
                
                self.list_of_clients[counter].setup_dataset(res)
        
                self.unique_labels[f"Client {counter+1}"] = {
                    "training" : training_data["unique_counter"], 
                    "validation_iid" : validation_data_iid["unique_counter"],
                    "validation_niid" : validation_data_niid["unique_counter"]
                    }
                
                counter+=1
                
                self.validation_results_niid=[
                    [
                        "client_id", "accuracy", "precision","recall", "aleatoric", "confidence", 
                        "mi", "correctness", "entropy", "variability", "loss", "ece", "mce"
                    ]
                ]
        else: 
            self.validation_datasets = datasource.get_validation_data()
            
            counter=0
            for training_data, validation_data in zip(self.training_datasets, self.validation_datasets):
                res = []
                [res.append(x) for x in training_data["unique_labels"] if x not in res]
                [res.append(x) for x in validation_data["unique_labels"] if x not in res]
                
                self.list_of_clients[counter].setup_dataset(res)
        
                self.unique_labels[f"Client {counter+1}"] = {
                    "training" : training_data["unique_counter"], 
                    "validation" : validation_data["unique_counter"]
                    }
                
                counter+=1
        
        self.testing_dataset = datasource.testset
            
        with open(f'{self.path}/Server_Data_distribution.yaml', 'w+') as outfile:
            yaml.dump(self.unique_labels, outfile)
        
    def load_from_checkpoit(self):
        self.model.load_state_dict(torch.load(f"{self.path}/Server_model.pt"))
        self.send_parameters()
        
        for i in range(0, self.workers):
            client = self.list_of_clients[i]
            client.training_performance = pd.read_csv(f"{self.path}/Client_{client.client_id}_training_performance.csv", index_col=0).values
            client.training_performance = client.training_performance.reshape(len(client.training_performance)).tolist()
            client.validation_performance = pd.read_csv(f"{self.path}/Client_{client.client_id}_validation_performance.csv", index_col=0).values.tolist()
            client.testing_performance = pd.read_csv(f"{self.path}/Client_{client.client_id}_testing_performance.csv", index_col=0).values.tolist()
            
            client.gradients = pd.read_csv(f"{self.path}/Client_{client.client_id}_gradients.csv", index_col=0).values.tolist()
            client.distributions_mean = torch.load(f'{self.path}/Client_{client.client_id}_weights.pt')
            try:
                client.cka = torch.load(f'{self.path}/Client_{client.client_id}_cka.pt')
            except FileNotFoundError:
                pass
            
            if client.is_adversary:
                try:
                    client.scores = pd.read_csv(f"{self.path}/Client_{client.client_id}_LO_scores.csv", index_col=0).values.tolist()
                except FileNotFoundError:
                    client.scores = pd.read_csv(f"{self.path}/Client_{client.client_id}_scores.csv", index_col=0).values
                    client.scores = client.scores.tolist()
                    
                client.hvs_list = pd.read_csv(f"{self.path}/Client_{client.client_id}_hvs_list.csv", index_col=0).values.tolist()
                client.lambdas = torch.load(f'{self.path}/Client_{client.client_id}_lambdas.pt')["lambdas"]
                if hasattr(client, "historical_data"):
                    try:
                        client.historical_data = torch.load(f'{self.path}/Client_{client.client_id}_historical_data.pt')["historical_data"]
                    except FileNotFoundError:
                        pass
                    
            if hasattr(client, "personalised_model") and self.aggregation=="Ditto":
                client.model_per.load_state_dict(torch.load(f"{os.getcwd()}/checkpoint/{self.server_config['name']}/Client_{client.client_id}_model_per.pt"))
        
        self.validation_results = pd.read_csv(f"{self.path}/Server_validation_performance.csv", index_col=0).values.tolist()
        self.testing_results = pd.read_csv(f"{self.path}/Server_testing_performance.csv", index_col=0).values.tolist()
        self.distributions_mean = torch.load(f'{self.path}/Server_weights.pt')
        
    def early_stopping_func(self, val_lost):
        delta = 0.25
        bench = self.previous_loss*delta
        if val_lost <= (self.previous_loss):
            self.chances+=1
            self.previous_loss = val_lost
            if self.chances >= self.patience:
                self.stop = True
        else:
            self.previous_loss = val_lost
            self.counter = 0
    
    def fit(self) -> None:
                
        if self.checkpoint:
            self.load_from_checkpoit()
            training_round = int(np.loadtxt(f"{self.path}/epoch.txt"))
        else:
            training_round = 0
        
        for j in range(training_round, self.rounds):
            print(f"Round {j+1:3} / {self.rounds}")
            logger.info('==================\nStart of Round {:3}\n================\n'.format(j+1))
            weights = [] if self.aggregation in self.available_aggregation else {}
            
            for i in range(0, self.workers):
                """Get Clients and Datasets"""
                client = self.list_of_clients[i]
                training_dataset = self.training_datasets[i]
                
                """Train the Client"""
                if client.trainable:
                    client.train(training_dataset, epoch=j)
                    client.model = client.model.to('cpu')
                    weights.append(copy.deepcopy(client.model.state_dict()))
            
            #TODO: Need to find a space where to add the client selection algorithm
            clients_weights = copy.deepcopy(weights)
            """Aggregation"""
            self.global_weight = copy.deepcopy(self.aggregator(copy.deepcopy(clients_weights)))
            if self.global_weight is not None:
                self.model.load_state_dict(copy.deepcopy(self.global_weight))
                
            logger.info('GPU Temperature {}'.format(get_gpu_temperature()))
            
            logger.info('==================\Model Evaluation\n================\n')
            
            """Model Evaluation"""
            self.global_evaluation()
            self.personalised_evaluation()
            
            if self.server_config["learning_config"]["uncertainty_evaluation"]:
                self.uncertainty_evaluation()
            
            self.graph_analysis()
            
            self.send_parameters()
            
            weights, _, _ = read_weights(self.model, self.server_config)
        
            self.distributions_mean.append(weights)
            
            """Checkpoint"""
            if j < (self.rounds-1):
                self.save_checkpoint(j)
            
            try:
                commit_results(self.workers, f"{self.path}")
            except:
                logger.error('Error with checkpoint')
            
            try:
                commit_to_git()
                
            except:
                logger.error('Error with the Git')
            
            if self.stop:
                break
            
        self.log_results()
        
        try:
            commit_to_git()
        except:
            logger.error('Error with the Git')
            
        dataset = self.server_config["collection"]["selection"]
            
        torch.save(self.model.state_dict(), f"{self.server_config['name']}_Server_{dataset}_{self.seed}.pt")
        
        
    def send_parameters(self):
        for client in self.list_of_clients:
            client.set_parameters(self.model)
            
    def select_clients(self):
        pass
                       
    def aggregator(self, weights) -> dict:
        if self.aggregation == "FedAvg":
            return FedAvg(copy.deepcopy(weights))
        elif self.aggregation == "Krum":
            if not isinstance(weights, dict):
                dict_weights = {idx : weight for idx, weight in enumerate(weights)}
            return Krum(dict_weights, self.workers)
        elif self.aggregation == "FedDyn": 
            self.update_server_state()
            self.aggregate_parameters()
        elif self.aggregation == "Ditto": 
            return FedAvg(copy.deepcopy(weights))
        elif self.aggregation == "pFedME": 
            return PFedAvg(copy.deepcopy(self.model),weights)
        elif self.aggregation == "PerFedAvg": 
            return FedAvg(copy.deepcopy(weights))
        elif self.aggregation == "FedProx": 
            return FedAvg(copy.deepcopy(weights))
        
    def evaluation(self, **kwargs):
        if 'validation' in kwargs:
            dataset = kwargs['validation']
            dataset = dataset['dataloader']
        elif "validation_iid" in kwargs:
            dataset = kwargs['validation_iid']
            dataset = dataset['dataloader']
        elif "validation_niid" in kwargs:
            dataset = kwargs['validation_niid']
            dataset = dataset['dataloader']  
        else:
            dataset = kwargs['testing']
            
        accuracy, precision, recall, aleatoric, confidence = 0, 0, 0, 0, 0
        mi, correctness, entropy, variability, matrix, loss = 0, 0, 0, 0, 0,0
        
        self.model.eval()                
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataset):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    self.model= self.model.to(self.device)
                    output = self.model(data)
                    
                    loss_func = self.loss_function(output, target)
                    
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
                
        self.model = self.model.to('cpu')
            
        result = [
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
            loss,
            results["ece"],
            results["mce"]
        ]
        
        if 'validation' in kwargs or 'validation_iid' in kwargs:
            self.validation_results.append(result)
        elif 'validation_niid' in kwargs:
            self.validation_results_niid.append(result)
        else:
            indices = self.get_best_indices()   
            self.best_indices_arr.append(indices)
            self.testing_results.append(result)

            if self.early_stopping:
                self.early_stopping_func(loss)
                
        

    def global_evaluation(self, **kwargs):
        self.evaluation(testing = self.testing_dataset)
        
        for i in range(0, self.num_workers):
            self.list_of_clients[i].testing(self.testing_dataset)
            
            if hasattr(self.list_of_clients[i], 'personalised_model'):
                self.list_of_clients[i].per_testing(self.testing_dataset)
        print('')
        
    def personalised_evaluation(self, **kwargs):
        for i in range(0, self.num_workers):
            if hasattr(self,"validation_datasets_iid"):         
                self.evaluation(validation_iid = self.validation_datasets_iid[i])
                self.evaluation(validation_niid = self.validation_datasets_niid[i])
                
                self.list_of_clients[i].validation(self.validation_datasets_iid[i], niid=False)
                self.list_of_clients[i].validation(self.validation_datasets_niid[i], niid=True)
                
                if hasattr(self.list_of_clients[i], 'personalised_model'):
                    # self.list_of_clients[i].per_validation(self.validation_datasets_iid[i])#, niid=False)
                    self.list_of_clients[i].per_validation(self.validation_datasets_niid[i])#, niid=True)
            else:
                self.evaluation(validation = self.validation_datasets[i])
                self.list_of_clients[i].validation(self.validation_datasets[i])
                
                if hasattr(self.list_of_clients[i], 'personalised_model'):
                    self.list_of_clients[i].per_validation(self.validation_datasets[i])
        print('')
    
    def uncertainty_evaluation(self):
        # self.global_uncertainty_evaluation(self.training_datasets[-1], self.validation_datasets_iid[-1], self.testing_dataset)
        
        for i in range(0, self.num_workers):
            self.list_of_clients[i].uncertainty_evaluation(self.training_datasets[i],self.validation_datasets[i],self.testing_dataset)
    
    # def global_uncertainty_evaluation(self, training_dataset, validation_dataset, testing_dataset):
    #     scores = federated_model_analysis(
    #         server_model=self.model,
    #         client_models=[client.model for client in self.list_of_clients],
    #         test_data=testing_dataset,
    #         num_clusters=2
    #     )
        
    #     self.uncertainty_evaluation_perf.append(scores)

    def graph_analysis(self):
        
        client_models = {}
        for i in range(0, self.workers):
            client = copy.deepcopy(self.list_of_clients[i])

            weight , _, _ = read_weights(copy.deepcopy(client.model), self.server_config)
            
            client_models[client.client_id] = copy.deepcopy(weight)
        
        global_weight , _, _ = read_weights(copy.deepcopy(self.model), self.server_config)
        
        analysis = FLGraphAnalysis(
            global_model= copy.deepcopy(global_weight),
            client_models= copy.deepcopy(client_models),
            malicious_clients= self.adversarial_clients.tolist(),
            server_config= self.server_config,
            global_confidence=self.testing_results[-1][5],
            client_confidences={client.client_id: client.testing_performance[-1][5] for client in self.list_of_clients},
            best_indices=self.get_best_indices()
        )
        
        results = analysis.analyze()

        self.graph_analysis_results.append(results)
        self.graph_analysis_latex.append(
            nx.to_latex_raw(
                analysis.graph, 
                edge_options={'style': 'very thick', 'color': 'blue'},
                node_options={'shape': 'circle', 'fill': 'white'},
                edge_label_options={'fill': 'white'}
            )
        )
        self.graph_analysis_json.append(
            nx.readwrite.json_graph.adjacency_data(analysis.graph)
        )
    
    def log_results(self):
        dataset = self.server_config["collection"]["selection"]
        d = datetime.datetime.now()
        
        split_path = self.path.split("/")
        path = f"./results/runs/{d.year}{d.month if d.month>9 else f'0{d.month}'}{d.day if d.day>9 else f'0{d.day}'}/{split_path[-1]}"
        
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path)
        except:
            path="."
            
        model_type = self.server_config["learning_config"]["global_model"]['type']
        
        for client in self.list_of_clients:
            pd.DataFrame(client.training_performance).to_csv(f"{path}/Client_{client.client_id}_training_performance.csv")
            pd.DataFrame(client.validation_performance).to_csv(f"{path}/Client_{client.client_id}_validation_performance.csv")
            pd.DataFrame(client.testing_performance).to_csv(f"{path}/Client_{client.client_id}_testing_performance.csv")
            
            if hasattr(client, "validation_performance_niid"):
                pd.DataFrame(client.validation_performance_niid).to_csv(f"{path}/Client_{client.client_id}_validation_performance_niid.csv")

            if self.server_config["learning_config"]["uncertainty_evaluation"]:
                pd.DataFrame(client.uncertainty_evaluation_perf).to_csv(f"{path}/Client_{client.client_id}_uncertainty_evaluation.csv")
            
            pd.DataFrame(client.best_indices).to_csv(f"{path}/Client_{client.client_id}_best_indices.csv")
            pd.DataFrame(client.gradients).to_csv(f"{path}/Client_{client.client_id}_gradients.csv")
            
            torch.save(client.distributions_mean, f"{path}/Client_{client.client_id}_weights.pt")
            
            pd.DataFrame(client.cka).to_csv(f"{path}/Client_{client.client_id}_cka.csv")
            
            torch.save(client.model.state_dict(), f"{path}/Client_{client.client_id}_model.pt")
                    
            if client.is_adversary:
                torch.save(client.attack_engangement, f"{path}/Client_{client.client_id}_engangement.pt")
                pd.DataFrame(client.scores).to_csv(f"{path}/Client_{client.client_id}_LO_scores.csv")
                pd.DataFrame(client.hvs_list).to_csv(f"{path}/Client_{client.client_id}_hvs_list.csv")  
                torch.save(client.lambdas, f"{path}/Client_{client.client_id}_lambdas.pt")
                    
            if hasattr(client, 'personalised_model'):
                pd.DataFrame(client.per_validation_performance).to_csv(f"{path}/Client_{client.client_id}_per_validation_performance.csv")
                pd.DataFrame(client.per_testing_performance).to_csv(f"{path}/Client_{client.client_id}_per_testing_performance.csv")
                torch.save(client.personalised_model.state_dict(), f"{path}/Client_{client.client_id}_model.pt")
            
        pd.DataFrame(self.validation_results).to_csv(f"{path}/Server_validation_performance.csv")
        pd.DataFrame(self.testing_results).to_csv(f"{path}/Server_testing_performance.csv")
        torch.save(self.distributions_mean, f"{path}/Server_weights.pt")
        torch.save(self.graph_analysis_results, f"{path}/Server_graph_analysis.pt")
        torch.save(self.graph_analysis_latex, f"{path}/Server_graph_analysis_latex.pt")
        torch.save(self.graph_analysis_json, f"{path}/Server_graph_analysis_json.pt")
        
        torch.save(self.model.state_dict(), f"{path}/Server_model.pt")
        
        with open(f'{path}/config.yaml', 'w+') as outfile:
            yaml.dump(self.server_config, outfile)
        
        with open(f'{path}/Server_Data_distribution.yaml', 'w+') as outfile:
            yaml.dump(self.unique_labels, outfile)
            
        self.results_path = path
        
    def save_checkpoint(self, training_round: int) -> None:
        """Saves inside the define path located in self.path a checkpoint of the training

        Args:
            training_round (int): the current training round
            
        """
        for client in self.list_of_clients:
            pd.DataFrame(client.training_performance).to_csv(f"{self.path}/Client_{client.client_id}_training_performance.csv")
            pd.DataFrame(client.validation_performance).to_csv(f"{self.path}/Client_{client.client_id}_validation_performance.csv")
            pd.DataFrame(client.testing_performance).to_csv(f"{self.path}/Client_{client.client_id}_testing_performance.csv")
            
            if hasattr(client, "validation_performance_niid"):
                pd.DataFrame(client.validation_performance_niid).to_csv(f"{self.path}/Client_{client.client_id}_validation_performance_niid.csv")

            pd.DataFrame(client.gradients).to_csv(f"{self.path}/Client_{client.client_id}_gradients.csv")
            
            pd.DataFrame(client.cka).to_csv(f"{self.path}/Client_{client.client_id}_cka.csv")

            if self.server_config["learning_config"]["uncertainty_evaluation"]:
                try:
                    pd.DataFrame(client.uncertainty_evaluation_perf).to_csv(f"{self.path}/Client_{client.client_id}_uncertainty_evaluation.csv")
                except:
                    pass
            torch.save(client.distributions_mean, f"{self.path}/Client_{client.client_id}_weights.pt")
                
            if client.is_adversary:
                pd.DataFrame(client.scores).to_csv(f"{self.path}/Client_{client.client_id}_scores.csv")
                pd.DataFrame(client.hvs_list).to_csv(f"{self.path}/Client_{client.client_id}_hvs_list.csv")
                
                torch.save({"lambdas" : client.lambdas if len(client.lambdas) > 0 else "Empty"}, f'{self.path}/Client_{client.client_id}_lambdas.pt')
                
                if hasattr(client, "historical_data"):
                    torch.save({"historical_data" : client.historical_data if len(client.historical_data) > 0 else "Empty"}, f'{self.path}/Client_{client.client_id}_historical_data.pt')
                
        pd.DataFrame(self.validation_results).to_csv(f"{self.path}/Server_validation_performance.csv")
        pd.DataFrame(self.testing_results).to_csv(f"{self.path}/Server_testing_performance.csv")
        torch.save(self.distributions_mean, f"{self.path}/Server_weights.pt")
        torch.save(self.graph_analysis_results, f"{self.path}/Server_graph_analysis.pt")
        torch.save(self.graph_analysis_latex, f"{self.path}/Server_graph_analysis_latex.pt")
        torch.save(self.graph_analysis_json, f"{self.path}/Server_graph_analysis_json.pt")
        
        torch.save(self.model.state_dict(), f"{self.path}/Server_model.pt")
        
        np.savetxt(f"{self.path}/epoch.txt", [training_round+1])
        
        if not training_round > 1:
            with open(f'{self.path}/config.yaml', 'w+') as outfile:
                yaml.dump(self.server_config, outfile)
                
        with open(f'{self.path}/Server_Data_distribution.yaml', 'w+') as outfile:
            yaml.dump(self.unique_labels, outfile)
            
    def get_best_indices(self):
        weights, _, _ =  read_weights(self.model, self.server_config)
        
        all_indices = { key: [np.random.randint(0, len(weights[key])) for _ in range(0, self.server_config["attack"]["num_of_neurons"])] for key in weights.keys()}
        
        for key, indices in all_indices.items():
            while len(np.unique(indices)) != self.server_config["attack"]["num_of_neurons"]:
                indices = [ np.random.randint(0, len(weights[key])) for _ in range(0, self.server_config["attack"]["num_of_neurons"])]
        
        return all_indices