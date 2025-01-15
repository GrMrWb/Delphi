import copy
import logging
import torch
import torch.nn as nn
import yaml
from scipy.special import rel_entr
from scipy.stats import norm


from src.configLibrary import read_client_config, read_server_config, read_experiment_config
from src.learning.split_learning.models import AlexNet, AlexNetClient

from src.evaluation.utils import EvaluationMetrices
from src.delphi.strategy import *
from src.delphi.utils import (
    modify_the_weights_with_distrbution, 
    find_the_best_features, 
    get_weight_distribution,
    get_similarity_features,
    get_similarity_between_server
    )
from src.learning.utils import choose_loss_fn, choose_optimizer

from src.visualisation.utils import plot_t_sne

import torch.quantization
import torch.quantization._numeric_suite as ns
from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

class SplitTrainer:
    def __init__(self, client_id, client_model, 
                client_optimizer, config, server_model, 
                server_optimizer, dataset, 
                device, threat_model=None, adversary = False, 
                bo_x=None, bo_y=None, 
                best_features=None, prev_gradients=None, 
                weight_optimiser = None
            ):
        
        self.is_adversary = adversary
        self.client_id = client_id
        self.client_model = client_model
        self.client_optimizer = client_optimizer
        self.config = config
        
        if self.is_adversary:
            self.threat_model = threat_model
            self.optimisation = None
            self.bo_x = bo_x
            self.bo_y = bo_y
            self.weight_optimiser = weight_optimiser
            self.best_features = best_features
            self.prev_gradients = prev_gradients
            
        self.server_model = server_model
        self.server_optimizer = server_optimizer
        self.loss_fn = choose_loss_fn()
        self.dataset = dataset
        self.device = device
    
    def __normal_client__(self, data: torch.Tensor, target: torch.Tensor, batch_idx:int, size:int) -> float:
        self.client_model = self.client_model.to(self.device)
        self.server_model = self.server_model.to(self.device)
        data, target = data.to(self.device), target.to(self.device)
        self.client_optimizer.zero_grad()
        self.server_optimizer.zero_grad()
        
        cut_layer = self.client_model(data)
        
        input_for_server = cut_layer
        
        input_for_server = input_for_server.to(self.device)
        input_for_server.retain_grad()
        output = self.server_model(input_for_server)
        
        loss = self.loss_fn(output, target)
        loss.backward(retain_graph=True)
        
        cut_layer.backward(input_for_server.grad)

        self.client_optimizer.step()
        self.server_optimizer.step()
        
        print('\rClient {} | Training Loss: {:.5f} | Batch: {}/{}'.format(self.client_id, loss.item(), batch_idx, size), end='\r')
        
        del data, target
        
        return loss.item()
    
    def __adversarial_client__(self, data: torch.Tensor, target: torch.Tensor, epoch, batch_idx, size) -> None:
        enganging_criteria=0
        engage = (epoch > enganging_criteria)
        
        self.client_model = self.client_model.to(self.device)
        self.server_model = self.server_model.to(self.device)
        data, target = data.to(self.device), target.to(self.device)
        
        self.client_optimizer.zero_grad(), self.server_optimizer.zero_grad()
        
        if not engage:
            old_weights = copy.deepcopy(self.client_model.net.conv1.weight.data.cpu().detach())
            
            #Feed the models
            cut_layer_client_model = self.client_model(data)
        
            input_for_server = cut_layer_client_model
            input_for_server = input_for_server.to(self.device)
            input_for_server.retain_grad()
            
            output = self.server_model(input_for_server)
            
            #Backprop for the loss
            loss = self.loss_fn(output, target)
            loss.backward(retain_graph=True)
            _gradients = input_for_server.grad
            cut_layer_client_model.backward(_gradients)
            
            #Optimisation
            self.server_optimizer.step(), self.client_optimizer.step()
            
            if loss != None:
                best_features, self.prev_gradients, weights_after, weights_before = find_the_best_features(target, self.client_model, old_weights)
                best_weight_distributions = get_weight_distribution(self.client_model, best_features)
                
                self.best_features = [best_features] if epoch==0 and batch_idx==0 else np.concatenate((self.best_features, [best_features]))
            else:
                enganging_criteria+=1
                
            feature_similarity, weight_similarity = get_similarity_features(weights_after), get_similarity_between_server(weights_before, weights_after)
            
            self.threat_model.load_state_dict(self.client_model.state_dict())
                   
        else:
            new_distribution = self.weight_optimiser.run_BO(self.bo_x, self.bo_y)
            new_distribution = new_distribution.reshape(5,2)
            
            old_weights = copy.deepcopy(self.client_model.net.conv1.weight.data.cpu().detach())
            
            try:
                new_weights = modify_the_weights_with_distrbution(old_weights, self.best_features[-1], self.prev_gradients, new_distribution, self.device)
            except:
                return 10.0 , False, False
            self.threat_model.net.conv1.weight.data = new_weights
            
            self.threat_model = self.threat_model.to(self.device)
            
            # Feed models
            cut_layer_threat_model = self.threat_model(data)
            cut_layer_client_model = self.client_model(data)
            
            input_for_server = cut_layer_threat_model
            
            input_for_server = input_for_server.to(self.device)
            input_for_server.retain_grad()
            output = self.server_model(input_for_server)
            
            # Backprop for the loss
            try:
                loss = self.loss_fn(output, target)
                loss.backward(retain_graph=True)
                _gradients = input_for_server.grad
                cut_layer_client_model.backward(_gradients)
                cut_layer_threat_model.backward(_gradients)
                
            except:
                input_for_server = cut_layer_client_model
            
                input_for_server = input_for_server.to(self.device)
                input_for_server.retain_grad()
                output = self.server_model(input_for_server)
                
                loss = self.loss_fn(output, target)
                try:
                    loss.backward(retain_graph=True)
                except:
                    return loss.item(), False, False
            
            # Optimisation
            self.server_optimizer.step(), self.client_optimizer.step()
            
            #Find the best features and distributions
            best_features, self.prev_gradients, weights_after, weights_before = find_the_best_features(target, self.client_model, old_weights)
            
            if best_features[0] == 0:
                best_features = None
                best_features = self.best_features[-1]
            
            best_weight_distributions = get_weight_distribution(self.client_model, best_features)
            
            self.best_features = [best_features] if epoch==0 and batch_idx==0 else np.concatenate((self.best_features, [best_features]))
            
            #Calculate similarity
            feature_similarity, weight_similarity = get_similarity_features(weights_after), get_similarity_between_server(weights_before, weights_after)

            del cut_layer_threat_model
            similarities = weight_similarity
            
            print('\rClient {} | Training Loss: {:.5f} | Batch: {}/{}'.format(self.client_id, loss.item(), batch_idx, size), end='\r')
            
            return loss.item(), similarities, new_distribution
            
        similarities = weight_similarity
        
        print('\rClient {} | Training Loss: {:.5f} | Batch: {}/{}'.format(self.client_id, loss.item(), batch_idx, size), end='\r')
        
        del data, target, input_for_server, cut_layer_client_model
        
        return loss.item(), similarities, best_weight_distributions
    
    def train(self, epoch):
        self.client_model.train(), self.server_model.train()
        
        running_loss = []
        running_similarities = []
        
        size = int(len(self.dataset.dataset)/256)
        
        torch.cuda.empty_cache()
        for batch_idx, (data, target) in enumerate(self.dataset):
            
            if not self.is_adversary:
                loss = self.__normal_client__(data, target, batch_idx, size) 
            else:
                self.threat_model.train()
                loss, similarities, best_weight_distributions = self.__adversarial_client__(data, target, epoch, batch_idx, size)

                if not isinstance(similarities, bool):
                    try:    
                        self.bo_x = best_weight_distributions.reshape(1,10) if batch_idx==0 and epoch==0 else np.concatenate((self.bo_x, best_weight_distributions.reshape(1,10)))
                    except:
                        self.bo_x = best_weight_distributions.reshape(1,10) if batch_idx==0 and epoch==0 else np.concatenate((self.bo_x, best_weight_distributions.cpu().detach().numpy().reshape(1,10)))
                    
                    self.bo_y.append(similarities)
                    running_similarities.append(similarities)
                # self.bo_y = similarities.reshape(1,5,2) if batch_idx==0 and epoch==0 else np.concatenate((self.bo_y, similarities.reshape(1,1,5)))

                if batch_idx > 8 and epoch!=0:
                    break
            
            running_loss.append(loss) 
        
        print('')
        if self.is_adversary:
            self.threat_model = self.client_model if epoch==0 else self.threat_model
        
        if self.is_adversary:
            return running_loss, running_similarities, self.client_model, self.threat_model, self.client_optimizer, self.bo_x, self.bo_y, self.best_features, self.prev_gradients, self.weight_optimiser, self.server_model, self.server_optimizer
        else:
            return running_loss, self.client_model, self.client_optimizer, self.server_model, self.server_optimizer
    
    def test(self, dataset, epoch, epochs):
        loss=[]
        
        for data, target in dataset:
            self.client_model.eval() if not self.is_adversary else self.threat_model.eval()
            self.server_model.eval()
            
            if not self.is_adversary:
                self.client_model = self.client_model.to(self.device)
            else:
                self.threat_model = self.threat_model.to(self.device)
                
            self.server_model = self.server_model.to(self.device)
            data, target = data.to(self.device), target.to(self.device)
            
            cut_layer = self.client_model(data)
            output = self.server_model(cut_layer)
            
            out = self.loss_fn(output, target).item()
            
            loss.append(out)
            
            print('\rClient {} |  Testing Loss: {:.5f} | Epoch: {}/{}'.format(self.client_id, out,epoch+1, epochs), end='\r')
        
        print('\rClient {} |  Testing Loss: {:.5f} | Epoch: {}/{}'.format(self.client_id, out,epoch+1, epochs), end='\n')

        return sum(loss)/len(loss)