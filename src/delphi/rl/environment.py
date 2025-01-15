"""
Author: GrMrWb

Based on Clean RL PPO code
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import functools, os
import copy

from src.delphi.utils import (
    write_weights, 
    read_weights, 
    get_uncertainty, 
    modify_the_weights_with_single_neuron
)
from src.delphi.rl.reward import (
    calculate_reward,
    calculate_single_reward
)

import gymnasium as gym

from pettingzoo.utils.env import ParallelEnv

from gymnasium.spaces import Box


class DelphiEnvSingleRL(gym.Env):
    metadata = {"name": "DelphiEnvRL"}
    def __init__(self, config, model, indices, data, target):
        super().__init__()
        self.device = "cuda"
        self.experiment_config = config
        
        self.max_time_steps = self.experiment_config["attack"]["rl"]["time_steps"]
        # current_step in the dataset
        self.current_step = 0
        
        # Reward
        self.current_reward = 0
        self.reward = []
        
        # Defining the Agents
        self.num_of_agents = config["attack"]["num_of_neurons"] * (len(self.experiment_config["attack"]["layer_of_interest"]) if isinstance(self.experiment_config["attack"]["layer_of_interest"], list) else self.experiment_config["attack"]["layer_of_interest"])
        self.indices = indices
        
        # Dataset
        self.data, self.target = data, target
        
        # Model
        self.model = copy.deepcopy(model)
        self.past_weights = None
        
        weights, _ ,_ = read_weights(model, config)
        self.weights_init = copy.deepcopy(weights)
        # Action Space
        # ============================================
        # Action = layer parameters

        shape = 0
        
        for layer, indices in self.indices.items():
            for index in indices:
                shape += np.array(weights[layer][index].shape).prod()
        
        self.action_spaces = {
            "single": Box(
                low = -int(config["attack"]["bounds"]),
                high = int(config["attack"]["bounds"]),
                shape= (int(shape), )
            )
        }
        
        # Observation State Space
        # ============================================
        # State = 
        
        self.observation_spaces = {
            "single": Box(
                low = -1, 
                high = 1,
                shape= (int(2*shape+1),)
            )
        }
        
        self.num_steps = config["attack"]["rl"]["time_steps"]
        self.learning_rate = config["attack"]["rl"]["learning_rate"]
        self.gamma = config["attack"]["rl"]["gamma"]
        self.gae_lambda = config["attack"]["rl"]["gae_lambda"]
        self.batch_size = config["attack"]["rl"]["time_steps"]
        self.minibatch_size = self.batch_size / config["attack"]["rl"]["num_minibatches"]
        self.num_iterations = self.num_steps*10 / self.batch_size
        self.clip_coef = config["attack"]["rl"]["clip_coef"] 
        self.clip_vloss = config["attack"]["rl"]["clip_vloss"]
        self.ent_coef = config["attack"]["rl"]["ent_coef"]
        self.vf_coef = config["attack"]["rl"]["vf_coef"]
        self.anneal_lr = config["attack"]["rl"]["anneal_lr"]
        self.update_epochs = config["attack"]["rl"]["update_epochs"]
        self.norm_adv = config["attack"]["rl"]["norm_adv"]
        self.max_grad_norm = config["attack"]["rl"]["max_grad_norm"]
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
       
    def reset(self, seed=None, options=None):
        # Select random steps from the dataset. Total Time steps in the dataset - 2 X Delay in the state
        # self.dataset = read_pre_generated_data()
        
        self.current_step = 0
        torch.cuda.empty_cache()
        self.current_reward = 0
        
        self.reward = {}
        
        obs = self._get_observation().reshape(1,-1)
        
        uncertainty = torch.tensor(float(get_uncertainty(self.model, self.data, self.target, self.experiment_config)))
        uncertainty = uncertainty.reshape(-1, 1)
        
        action = torch.zeros(self.action_spaces["single"].shape).unsqueeze(0)
        
        observations = torch.cat((uncertainty , obs, action) , dim=1)
        
        if self.past_weights==None:
            self.past_weights = observations
        
        return observations, ""

    def step(self, actions:dict):
        """
        Args:
            actions (dict): action for each agent

        Returns:
            tuple[dict, dict, dict, dict, dict]: observations, rewards, terminations, truncations, infos
        """        
        terminations = 1 if self.current_step > self.max_time_steps else 0

        self.current_step+=1
        
        env_truncation = self.current_step >= self.max_time_steps
        truncations = 0
        
        reward, uncertainty = calculate_single_reward(self.model, self.data, self.target, actions, self.indices, "cpu", self.experiment_config)
        
        infos = {
            "uncertainty" : uncertainty
        } 
        
        weights, _ ,_ = read_weights(self.model, self.experiment_config)

        candidates = []
        begin = 0
        for layer, values in self.indices.items():
            for index in values:
                end = begin + np.array(weights[layer][index].shape).prod()
                candidates.append(actions[:, begin:end])
                begin = end        

        counter=0
        new_model = copy.deepcopy(self.model)
        for layer, indices in self.indices.items():
            for index in indices:
                new_model = modify_the_weights_with_single_neuron(copy.deepcopy(new_model), candidates[counter], self.device, index, layer=layer, server_config=self.experiment_config)
                new_model = copy.deepcopy(new_model)
                counter+=1
            
        # keep track the real step because current step is based on the dataset
        uncertainty = self._get_uncertainty(actions)
        obs = self._get_observation().reshape(1, -1)
        uncertainty = uncertainty.reshape(1, -1)

        if torch.isnan(uncertainty):
            uncertainty = torch.tensor(1).reshape(1, -1)
            reward = torch.tensor(-100)
        
        observations = torch.cat((uncertainty , obs, actions.cpu()) , dim=1)

        return observations, reward, terminations, truncations, infos
        
    def _get_observation(self, step=0):
        
        weights, _ , _ = read_weights(self.model, self.experiment_config)
        
        obs_weights = None
        
        for layer, indices in self.indices.items():
            for index in indices:
                obs_weights = weights[layer][index].detach().cpu().flatten() if obs_weights == None else torch.cat((obs_weights , weights[layer][index].detach().cpu().flatten()) ,dim=0)
        
        return obs_weights
    
    def _get_uncertainty(self, new_weights):
        weights, _ ,_ = read_weights(self.model, self.experiment_config)
        
        candidates = []
        begin = 0
        for layer, values in self.indices.items():
            for index in values:
                end = begin + np.array(weights[layer][index].shape).prod()
                candidates.append(new_weights[:, begin:end])
                begin = end    
        
        counter = 0
        
        for layer, indices in self.indices.items():
            for index in indices:
                new_model = modify_the_weights_with_single_neuron(copy.deepcopy(self.model), candidates[counter], self.device, index, layer=layer, server_config=self.experiment_config)
                counter+=1
            
        uncertainties = float(get_uncertainty(new_model, self.data, self.target, self.experiment_config))
        
        if uncertainties == np.nan:
            uncertainties = 10
            
        return torch.tensor(uncertainties)
    

class DelphiEnvMultiRL(ParallelEnv):
    metadata = {"name": "DelphiEnvRL"}
    def __init__(self, config, model, indices, data, target):
        super().__init__()
        self.device = "cuda"
        self.experiment_config = config
        
        self.max_time_steps = self.experiment_config["attack"]["rl"]["time_steps"]
        # current_step in the dataset
        self.current_step = 0
        
        # Reward
        self.current_reward = 0
        self.reward = []
        
        # Defining the Agents
        self.num_of_agents = config["attack"]["num_of_neurons"]
        self.indices = indices
        self.agents = [f"agent_{i}" for i in range(self.num_of_agents)]
        self.possible_agents  = self.agents
        
        # Dataset
        self.data, self.target = data, target
        
        # Model
        self.model = copy.deepcopy(model)
        self.past_weights = None
        
        weights, _ ,_ = read_weights(model, config)
        
        # Action Space
        # ============================================
        # Action = layer parameters
        
        self.action_spaces = {
            agent : Box(
                low = -config["attack"]["bounds"],
                high = config["attack"]["bounds"],
                shape= (np.array(weights[0].shape).prod(), ),
            ) for agent in self.agents 
        }
        
        # Observation State Space
        # ============================================
        # State = [Imaginary or real, Antennas, Users]
        
        self.observation_spaces = {
            agent : Box(
                low = -1, 
                high = 1,
                shape= (np.array(weights[0].shape).prod()+1,) 
            ) for agent in self.agents 
        }
        
        self.num_steps = config["attack"]["rl"]["time_steps"]
        self.learning_rate = config["attack"]["rl"]["learning_rate"]
        self.gamma = config["attack"]["rl"]["gamma"]
        self.gae_lambda = config["attack"]["rl"]["gae_lambda"]
        self.batch_size = config["attack"]["rl"]["time_steps"]
        self.minibatch_size = self.batch_size / config["attack"]["rl"]["num_minibatches"]
        self.num_iterations = self.num_steps*5 / self.batch_size
        self.clip_coef = config["attack"]["rl"]["clip_coef"] 
        self.clip_vloss = config["attack"]["rl"]["clip_vloss"]
        self.ent_coef = config["attack"]["rl"]["ent_coef"]
        self.vf_coef = config["attack"]["rl"]["vf_coef"]
        self.anneal_lr = config["attack"]["rl"]["anneal_lr"]
        self.update_epochs = config["attack"]["rl"]["update_epochs"]
        self.norm_adv = config["attack"]["rl"]["norm_adv"]
        self.max_grad_norm = config["attack"]["rl"]["max_grad_norm"]
     
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
       
    def reset(self, seed=None, options=None):
        # Select random steps from the dataset. Total Time steps in the dataset - 2 X Delay in the state
        # self.dataset = read_pre_generated_data()
        
        self.current_step = 0
        torch.cuda.empty_cache()
        self.current_reward = 0
        
        self.reward = {}
        
        obs = self._get_observation()
        
        uncertainty = torch.tensor([float(get_uncertainty(self.model, self.data, self.target, self.experiment_config)) for _ in self.possible_agents])
        uncertainty = uncertainty.reshape(-1, 1)
        
        observations = torch.cat((uncertainty , obs) , dim=1)
        
        return observations, ""

    def step(self, actions:dict):
        """
        Args:
            actions (dict): action for each agent

        Returns:
            tuple[dict, dict, dict, dict, dict]: observations, rewards, terminations, truncations, infos
        """        
        terminations = [1 if self.current_step > self.max_time_steps else 0 for _ in self.agents]

        self.current_step+=1
        
        env_truncation = self.current_step >= self.max_time_steps
        truncations = [0 for _ in self.agents]
        
        rewards, uncertainty = calculate_reward(self.model, self.data, self.target, actions, self.indices, "cpu", self.experiment_config)
        
        infos = {
            agent: {
                # "length" :  self.current_step,
                "uncertainty" : uncertainty
            } 
            for agent in self.agents
        }
        
        counter=0
        for index in self.indices:
            new_model = modify_the_weights_with_single_neuron(self.model, actions[counter], self.device, index, server_config=self.experiment_config)
            self.model = copy.deepcopy(new_model)
            counter+=1
            
        # keep track the real step because current step is based on the dataset
        uncertainty = self._get_uncertainty(actions)
        obs = self._get_observation()
        uncertainty = uncertainty.reshape(-1, 1)
    
        observations = torch.cat((uncertainty , obs) , dim=1)

        return observations, rewards, terminations, truncations, infos
        
    def _get_observation(self, step=0):
        
        weights, _ , _ = read_weights(self.model, self.experiment_config)
        
        obs_weights = torch.tensor([weights[idx].detach().cpu().flatten().numpy() for idx in self.indices])
        
        return obs_weights
    
    def _get_uncertainty(self, new_weights):
        
        uncertainties = []
        counter=0
        for index in self.indices:
            new_model = modify_the_weights_with_single_neuron(self.model, new_weights[counter], self.device, index, server_config=self.experiment_config)
            counter+=1
            uncertainties.append(float(get_uncertainty(new_model, self.data, self.target, self.experiment_config)))
            
        return torch.tensor(uncertainties)
