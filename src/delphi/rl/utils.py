"""
Author: GrMrWb

Based on Clean RL PPO code
"""   

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from src.delphi.rl.agent import Agent
from src.delphi.rl.environment import DelphiEnvMultiRL, DelphiEnvSingleRL

from src.delphi.utils import (
    write_weights, 
    read_weights, 
    get_uncertainty, 
    modify_the_weights_with_single_neuron
)

from torch.utils.tensorboard import SummaryWriter

def encode_information(agents:object):
    """Encoding the information in order to use it later

    Args:
        agents (object): all the agents

    Returns:
        dict:
    """
    
    information = {}
    
    for agent in agents:
        information[agent] = {
            "reward" : [],
            "weights" : [],
            "uncertainty" : []
        }
    
    return information 
    
def run_single_ppo(config, model, indices, data, target):
    env = DelphiEnvSingleRL(config, model, indices, data, target)
    agent = Agent(env, "single")
        
    model, uncertainty, agent = single_ppo_training_loop(config, env, agent)
    
    return model, uncertainty
    
def single_ppo_training_loop(config, env, agent):
    writer = SummaryWriter()
    
    next_obs, _ = env.reset(seed=21325)
    # information = encode_information(env.possible_agents)
    
    obs = torch.zeros((env.num_steps,  np.array(env.observation_spaces["single"].shape).prod())).to(env.device)
    actions = torch.zeros((env.num_steps,np.array(env.action_spaces["single"].shape).prod())).to(env.device)
    logprobs = torch.zeros((env.num_steps)).to(env.device)
    rewards = torch.zeros((env.num_steps)).to(env.device)
    dones = torch.zeros((env.num_steps)).to(env.device)
    values = torch.zeros((env.num_steps)).to(env.device)
    advantages = torch.zeros_like(rewards).to(env.device)
    
    next_obs = torch.Tensor(next_obs).to(env.device)
    next_done = torch.zeros(1).to(env.device)
    for iteration in range(1, int(env.num_iterations) + 1):
        if env.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / env.num_iterations
            lrnow = frac * env.learning_rate

            agent.optimizer.param_groups[0]["lr"] = lrnow
            
        uncertainty = 0
        for step in range(config["attack"]["rl"]["time_steps"]):
            torch.cuda.empty_cache()
            obs[step] = next_obs
            dones[step] = next_done
            print(f"\rStep {step+1:03}/{config['attack']['rl']['time_steps']:03} | Iteration {iteration:06} | Uncertainty {uncertainty}", end="\r")
            
            next_obs = next_obs.to(env.device)
            
            with torch.no_grad():   
                action, logprob, _, value = agent.policy.get_action_and_value(next_obs)
                action = (0.25*((action - action.min()) / (action.max() - action.min())) -0.125)
                values[step] = value.flatten()
            
            actions[step] = action.cpu()
            logprobs[step] = logprob
            
            next_obs, reward, terminations, truncations, infos = env.step(action)
            
            next_done = np.logical_or(terminations, truncations)
            next_done = torch.tensor(next_done).to(env.device)
            rewards[step] = reward
            next_obs = torch.tensor(next_obs).to(env.device)
            
            uncertainty = infos["uncertainty"]
            
        with torch.no_grad():
            rewards = 3*((rewards - rewards.min())/(rewards.max()-rewards.min()))-1
            next_value = agent.policy.get_value(next_obs).reshape(1, -1)
            lastgaelam = 0
            for t in reversed(range(env.num_steps)):
                if t == env.num_steps - 1:
                    nextnonterminal = 1.0 - int(next_done)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + env.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + env.gamma * env.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
    
        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_spaces[agent.agent_id].shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_spaces[agent.agent_id].shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(env.batch_size)
        clipfracs = []
        for epoch in range(env.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, env.batch_size, int(env.minibatch_size)):
                end = start + int(env.minibatch_size)
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.policy.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > env.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if env.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - env.clip_coef, 1 + env.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if env.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -env.clip_coef,
                        env.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - env.ent_coef * entropy_loss + v_loss * env.vf_coef

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.policy.parameters(), env.max_grad_norm)
                agent.optimizer.step()

            # if env.target_kl is not None and approx_kl > env.target_kl:
            #     break
            
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], iteration)
        writer.add_scalar("losses/value_loss", v_loss.item(), iteration)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), iteration)
        writer.add_scalar("losses/entropy", entropy_loss.item(), iteration)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iteration)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), iteration)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), iteration)
        writer.add_scalar("losses/explained_variance", explained_var, iteration)
   
    agent.policy.eval()
    actions,_,_,_ = agent.policy.get_action_and_value(env.past_weights.cuda())
    
    weights, _ ,_ = read_weights(env.model, env.experiment_config)

    
    candidates = []
    begin = 0
    for layer, values in env.indices.items():
        for index in values:
            end = begin + np.array(weights[layer][index].shape).prod()
            candidates.append(actions[:, begin:end])
            begin = end   
   
    counter=0
    for layer, indices in env.indices.items():
        for index in indices:
            new_model = modify_the_weights_with_single_neuron(copy.deepcopy(env.model), candidates[counter], env.device, index, layer=layer, server_config=env.experiment_config)
            model = copy.deepcopy(new_model)
            counter+=1 
   
    return model, infos["uncertainty"], agent

def run_multi_ppo(config, model, indices, data, target):
    env = DelphiEnvMultiRL(config, model, indices, data, target)
    agents = [ Agent(env, f"agent_{idx}") for idx in range(config["attack"]["num_of_neurons"])]
        
    model, uncertainty = multi_ppo_training_loop(config, env, agents)
    
    return model, uncertainty
    
def multi_ppo_training_loop(config, env, agents):
    writer = SummaryWriter()
    
    next_obs, _ = env.reset(seed=21325)
    information = encode_information(env.possible_agents)
    
    obs = torch.zeros((env.num_steps, len(env.possible_agents),  np.array(env.observation_spaces["agent_1"].shape).prod())).to(env.device)
    actions = torch.zeros((env.num_steps, len(env.possible_agents), np.array(env.action_spaces["agent_1"].shape).prod())).to(env.device)
    logprobs = torch.zeros((env.num_steps, len(env.possible_agents))).to(env.device)
    rewards = torch.zeros((env.num_steps, len(env.possible_agents))).to(env.device)
    dones = torch.zeros((env.num_steps, len(env.possible_agents))).to(env.device)
    values = torch.zeros((env.num_steps, len(env.possible_agents))).to(env.device)
    
    next_obs = torch.Tensor(next_obs).to(env.device)
    next_done = torch.zeros(len(env.possible_agents)).to(env.device)
    
    for iteration in range(1, int(env.num_iterations) + 1):
        if env.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / env.num_iterations
            lrnow = frac * env.learning_rate
            for agent in agents:
                agent.optimizer.param_groups[0]["lr"] = lrnow
            
        for step in range(config["attack"]["rl"]["time_steps"]):
            torch.cuda.empty_cache()
            obs[step] = next_obs
            dones[step] = next_done
            print(f"\rStep {step+1:03}/{config['attack']['rl']['time_steps']:03} | Iteration {iteration:06}", end="\r")
            
            lambdas = torch.tensor(np.zeros_like(np.array([env.action_space(agent_id).sample() for agent_id in env.possible_agents])))
            next_obs = next_obs.to(env.device)
            
            for idx, agent in enumerate(agents):
                with torch.no_grad():
                    action, logprob, _, value = agent.policy.get_action_and_value(next_obs[idx])
                    values[step][idx] = value.flatten()
                    lambdas[idx] = action.cpu()
                actions[step][idx]= action.cpu()
                logprobs[step][idx] = logprob
            
            next_obs, reward, terminations, truncations, infos = env.step(lambdas)
            
            next_done = np.logical_or(terminations, truncations)
            next_done = torch.tensor(next_done).to(env.device)
            for agent in range(len(env.possible_agents)):
                rewards[step][agent] = reward[agent]
                next_obs[agent] = torch.tensor(next_obs[agent]).to(env.device)
            
        for idx, agent in enumerate(agents):
            with torch.no_grad():
                next_value = agent.policy.get_value(next_obs[idx]).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(env.device)
                lastgaelam = 0
                for t in reversed(range(env.num_steps)):
                    if t == env.num_steps - 1:
                        nextnonterminal = 1.0 - int(next_done[idx])
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1][idx]
                        nextvalues = values[t + 1][idx]
                    delta = rewards[t][idx] + env.gamma * nextvalues * nextnonterminal - values[t][idx]
                    advantages[t][idx] = lastgaelam = delta + env.gamma * env.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
                
        for idx, agent in enumerate(agents):
            # flatten the batch
            b_obs = obs[:,idx]#.reshape((-1,) + np.prod(env.observation_spaces["agent_1"].shape))
            b_logprobs = logprobs[:,idx].reshape(-1)
            b_actions = actions[:,idx]#.reshape((-1,) + np.prod(env.action_spaces["agent_1"].shape))
            b_advantages = advantages[:,idx].reshape(-1)
            b_returns = returns[:,idx].reshape(-1)
            b_values = values[:,idx].reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(env.batch_size)
            clipfracs = []
            for epoch in range(env.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, env.batch_size, int(env.minibatch_size)):
                    end = start + int(env.minibatch_size)
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.policy.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > env.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if env.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - env.clip_coef, 1 + env.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if env.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -env.clip_coef,
                            env.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - env.ent_coef * entropy_loss + v_loss * env.vf_coef

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.policy.parameters(), env.max_grad_norm)
                    agent.optimizer.step()

                # if env.target_kl is not None and approx_kl > env.target_kl:
                #     break
                
                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                
            writer.add_scalar(f"{agent.agent_id}charts/learning_rate", agent.optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar(f"{agent.agent_id}losses/value_loss", v_loss.item(), iteration)
            writer.add_scalar(f"{agent.agent_id}losses/policy_loss", pg_loss.item(), iteration)
            writer.add_scalar(f"{agent.agent_id}losses/entropy", entropy_loss.item(), iteration)
            writer.add_scalar(f"{agent.agent_id}losses/old_approx_kl", old_approx_kl.item(), iteration)
            writer.add_scalar(f"{agent.agent_id}losses/approx_kl", approx_kl.item(), iteration)
            writer.add_scalar(f"{agent.agent_id}losses/clipfrac", np.mean(clipfracs), iteration)
            writer.add_scalar(f"{agent.agent_id}losses/explained_variance", explained_var, iteration)

    return env.model, infos["agent_1"]["uncertainty"]