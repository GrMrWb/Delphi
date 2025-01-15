import torch
import networkx as nx
import numpy as np
import re
from typing import List, Dict, Tuple
from scipy.stats import wasserstein_distance
import copy

from sklearn.preprocessing import MinMaxScaler
from src.delphi.utils.weight_io import read_weights

class FLGraphAnalysis:
    def __init__(self, 
                 global_model: torch.nn.Module, 
                 client_models: Dict[int, torch.nn.Module],
                 malicious_clients: List[int], 
                 server_config: Dict,
                 global_confidence: float = 0.95,
                 client_confidences: Dict[int, float] = None,
                 similarity_threshold: float = 0.5,
                 edge_weight_alpha: float = 1,
                 best_indices: dict = []):
        self.global_model = global_model
        self.client_models = client_models
        self.malicious_clients = malicious_clients if malicious_clients != [] else [1]
        self.server_config = server_config
        self.global_confidence = global_confidence
        self.client_confidences = {i: 0.25 for i in client_models.keys()} if not isinstance(client_confidences, dict) else client_confidences
        self.similarity_threshold = similarity_threshold
        self.edge_weight_alpha = edge_weight_alpha
        self.graph = None
        self.weight_metrics = {}
        self.best_indices = best_indices

    def remove_digits(self, s):
        return re.sub(r'\d+', '', s)

    def read_weights(self, model):
        """Read weights using the provided function"""
        configuration = self.server_config
        layers = configuration["attack"]["layer_of_interest"]
        strings = ["bn", "norm", "attn", "bias", "reduction"]
        multi_modality = False if configuration["learning_config"]["modality"] == "single" else True
        modal = f'{configuration["attack"]["modality"]}_encoder'
        layers = layers if isinstance(layers, list) else [layers]
        
        parameters = list(model.named_parameters())
        layers_named_weights = []
        type_of_layer = []
        
        for parameter in parameters:
            name = parameter[0].split(".")
            word_count = sum(
                1 for word in name
                if any(self.remove_digits(word) == self.remove_digits(s) for s in strings)
            )
            if word_count == 0 and "weight" in name:
                if len(parameter[1].shape) > 1:
                    layers_named_weights.append(parameter)
                    if multi_modality:
                        if not modal in name:
                            layers_named_weights.pop()
        
        weights, gradients = {}, {}
        for layer in layers:
            index = layer-1
            name_list = layers_named_weights[index][0].split(".")
            temp = model
            for name in name_list:
                temp = getattr(temp, name)
            weights[str(index)] = temp.data
            gradients[str(index)] = temp.grad.data if temp.grad is not None else 0
            type_of_layer.append(layers_named_weights[index][0])
            
        best_weights = {key : [weights[key][index].flatten().detach().cpu().numpy() for index in value] for key, value in self.best_indices.items()}
        
        return best_weights, gradients, type_of_layer

    def extract_layer_statistics(self, weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Extract statistics from weights using the new weight format"""
        stats = {}
        for layer_idx, weight in weights.items():
            stats[layer_idx] = []
            fmax = np.mean([weig.max() for weig in weight])
            for key, weig in enumerate(weight):
                weight_np = np.array(weig)/fmax #.cpu().numpy()
                stats[layer_idx].append({
                    'mean': np.mean(weight_np),
                    'std': np.std(weight_np),
                    'skew': float(np.mean(((weight_np - np.mean(weight_np)) / np.std(weight_np)) ** 3)),
                    'kurtosis': float(np.mean(((weight_np - np.mean(weight_np)) / np.std(weight_np)) ** 4) - 3),
                    'l1_norm': np.linalg.norm(weight_np.flatten(), ord=1),
                    'l2_norm': np.linalg.norm(weight_np.flatten(), ord=2)
                })
        return stats

    def calculate_weight_signature(self, model: torch.nn.Module) -> Dict[str, np.ndarray]:
        """Calculate weight signature using the new weight reading function"""
        weights, _, _ = self.read_weights(model)
        stats = self.extract_layer_statistics(weights)
        signatures = {}
        
        for layer_idx in stats:
            signatures[layer_idx] = np.array([stats[layer_idx][key]['l2_norm'] for key in range(len(stats[layer_idx]))]).mean()
        
        return signatures
    
    def calculate_weight_differences(self, model_A, model_B):
        distances = []
        weights_A = copy.deepcopy(model_A)
        weights_B = copy.deepcopy(model_B)
        
        best_weights_A = {key : [weights_A[key][index].flatten().detach().cpu().numpy() for index in value] for key, value in self.best_indices.items()}
        best_weights_B = {key : [weights_B[key][index].flatten().detach().cpu().numpy() for index in value] for key, value in self.best_indices.items()}
        
        best_weights_A = weights_A
        best_weights_B = weights_B
        
        for layer_idx, weight_A in best_weights_A.items():
            weight_A = weight_A.flatten().detach().cpu().numpy()
            weight_B = best_weights_B[layer_idx].flatten().detach().cpu().numpy()
            # fmax_A = np.mean([weig.max() for weig in weight_A])
            # fmax_B = np.mean([weig.max() for weig in weight_B])
            for key, weig in enumerate(weight_A):
                weight_A_np = np.array(weig) #/fmax_A
                weight_B_np = np.array(weight_B[key]) #/fmax_B
                
                distance = weight_A_np - weight_B_np
                
                distances.append(distance)
        
        return np.linalg.norm(distances, 2)

    def calculate_model_distance(self, model1: torch.nn.Module, model2: torch.nn.Module) -> float:
        sig1 = self.calculate_weight_signature(model1)
        sig2 = self.calculate_weight_signature(model2)
        
        distances = []
        for layer_idx in sig1:
            if layer_idx in sig2:
                scaler = MinMaxScaler()
                sig1_norm = scaler.fit_transform(sig1[layer_idx].reshape(-1, 1)).ravel()
                sig2_norm = scaler.transform(sig2[layer_idx].reshape(-1, 1)).ravel()
                
                euclidean_dist = sig1[layer_idx].reshape(-1, 1) - sig2[layer_idx].reshape(-1, 1)
                # wasserstein_dist = wasserstein_distance(sig1[layer_idx] - sig2[layer_idx])
                combined_dist = (self.edge_weight_alpha * euclidean_dist
                            # + (1 - self.edge_weight_alpha) * wasserstein_dist
                            )
                distances.append(combined_dist)
        
        return np.linalg.norm(distances, 2) if distances else float('inf')

    def build_graph(self) -> nx.Graph:
        """Build the graph based on model similarities"""
        G = nx.Graph()
        G.add_node("server")
        
        for client1 in self.client_models:
            G.add_node(client1)
            for client2 in self.client_models:
                if not client1 == client2:  # Avoid duplicate edges
                    distance = self.calculate_weight_differences(
                        self.client_models[client1],
                        self.client_models[client2]
                    )
                    if distance < self.similarity_threshold and distance > 0.01:
                        # Use the average confidence of the two clients for the edge weight
                        avg_confidence = (self.client_confidences[client1] + 
                                       self.client_confidences[client2]) / 2
                        G.add_edge(client1, client2, weight=distance)
    
            distance = self.calculate_weight_differences(
                self.client_models[client1],
                self.global_model
            )
            if distance < self.similarity_threshold  and distance > 0.01:
                # Use the average confidence of the two clients for the edge weight
                avg_confidence = (self.client_confidences[client1] + 
                                self.global_confidence) / 2
                G.add_edge(client1, "server", weight=distance)
            
        self.graph = G
        return G

    def calculate_attack_effectiveness(self) -> float:
        """Calculate the effectiveness of the attack"""
        rho_G = 0
        for client in self.malicious_clients:
            diff = self.calculate_weight_differences(self.global_model, self.client_models[client] if client!=0 else self.client_models[client+1])
            rho_G += diff
        
        return rho_G / (self.global_confidence * len(self.malicious_clients))

    def calculate_upper_bound(self) -> float:
        """Calculate theoretical upper bound"""
        n = self.graph.number_of_nodes()
        d_max = max(dict(self.graph.degree()).values())
        lambda_1 = max(np.real(np.linalg.eigvals(nx.to_numpy_array(self.graph))))
        try:
            Q = nx.community.modularity(self.graph, nx.community.louvain_communities(self.graph))
        except:
            Q = 0
            
        k = 1  # This constant may need to be adjusted based on empirical observations
        avg_client_confidence = np.mean(list(self.client_confidences.values()))
        
        bound = ((1/avg_client_confidence) * k * (d_max / n) * lambda_1 * 
                np.log(1 + d_max) * (2 - Q) * (len(self.malicious_clients) / n))
        return bound, Q

    def analyze(self) -> Dict[str, float]:
        """Perform complete analysis"""
        if self.graph is None:
            self.build_graph()
            
        attack_effectiveness = self.calculate_attack_effectiveness()
        upper_bound, Q = self.calculate_upper_bound()
        
        return {
            "attack_effectiveness": attack_effectiveness,
            "upper_bound": upper_bound,
            "graph_density": nx.density(self.graph),
            "average_clustering": nx.average_clustering(self.graph),
            "modularity": Q
        }

# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    import torch.nn as nn
    import yaml, copy
    
    from src.learning.models import Resnet18
    base_model = Resnet18

    clients = {}
    client_conf = {}
    for i in range(10):
        clients[f"client_{i}"] = copy.deepcopy(base_model)
        clients[f"client_{i}"].load_state_dict(torch.load(f"results/runs/20241012/FL_FedAvg_Resnet18_CIFAR10_imbalanced_single_rl_kl_div_attackers_2_neurons_5_seed_21325/Client_{i+1}_model.pt"))
        client_conf[f"client_{i}"]= 0.30

    with open("results/runs/20241012/FL_FedAvg_Resnet18_CIFAR10_imbalanced_single_rl_kl_div_attackers_2_neurons_5_seed_21325/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    f.close()
    
    server_model = base_model
    server_model.load_state_dict(torch.load("results/runs/20241012/FL_FedAvg_Resnet18_CIFAR10_imbalanced_single_rl_kl_div_attackers_2_neurons_5_seed_21325/Server_model.pt"))

    # Initialize analyzer
    analysis = FLGraphAnalysis(
        global_model=server_model,
        client_models=clients,
        client_confidences=0.30,
        malicious_clients=["client_1","client_7"],
        server_config=config,
        # target_layers= [1, 2]
    )

    # Run analysis
    results = analysis.analyze()

    # Print results
    print("Analysis Results:")
    print(f"Attack Effectiveness: {results['attack_effectiveness']:.4f}")
    print(f"Theoretical Upper Bound: {results['upper_bound']:.4f}")
    print(f"Graph Density: {results['graph_density']:.4f}")
    print(f"Average Clustering: {results['average_clustering']:.4f}")
    print(f"Modularity: {results['modularity']:.4f}")