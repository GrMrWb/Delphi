import os, shutil
import torch
import datetime
import logging
import warnings
from importlib import reload
import config
import random
import argparse
import numpy as np
warnings.filterwarnings('ignore')

def run(seed: int, checkpoint: bool, checkpoint_path: str, nums: int) -> None:    
    print("\nOlentzas clear, clear to start", end='\n')
    
    configuration = config.server_configuration
    
    config.server_configuration["attack"]["num_of_neurons"] = nums
            
    d = datetime.datetime.now()
    
    reload(logging)
    
    logfile_name = f'./application_logs/{configuration["name"]}_{configuration["collection"]["selection"]}_{d.year}-{d.month}-{d.day}_{d.hour}-{d.minute}{d.second:02}.log'
    
    logging.basicConfig(
        filename=logfile_name,
        level=logging.INFO,
        filemode='w+',
        format="[%(levelname)s] (%(asctime)s)\n%(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p"
    )
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    logger.info(f"Configuration\n{configuration}")
    print('\rI have started configurating the server. Hold a second', end='\r')
    print('\r                                                      ', end='\r')
    
    from src.learning.server import Server
    from src.learning.utils import commit_to_git
    
    host_server = Server(configuration, seed, checkpoint, checkpoint_path=checkpoint_path)
    
    if not (configuration["configuration"]=="CML"):
        users = config.server_configuration["learning_config"]["K"]
    
        print('Server is ready to use!! \nWe have {} clients configured'.format(users), end='\n')

    else:
        print('Server is ready to use!! \nWe have the central ML client configured', end='\n')

    host_server.fit()

    print('\rGit Git Git Git Git Git', end='\r')
    
    try:    
        commit_to_git()
    except:
        pass
    
    print('All have been submitted in the github. Have a nice day \nJarvis the best robot in the universe ;)') 
    
    # logger.removeHandler(logger.handlers[0])
    logger.handlers.clear()
    
    shutil.move(logfile_name, host_server.results_path)
    
    
def main(args: object, prefix: str) -> None:
    
    checkpoint = not int(args.checkpoint) == 0
    iid = True if int(args.iid)==1 else False
    use_neurons = not int(args.neurons) == 0
    active_attack = True if int(args.active_attack) == 1 else False
    features_indices = "continues" if int(args.features_indices) == 0 else "fixed"
    use_logits = True if int(args.use_logits) == 1 else False
    mp = not int(args.multiprocessing) == 0
    
    experiment = f"config/experiments/{args.dataset}/{prefix}_{args.aggregation}_{args.model}.yaml"
    
    reload(config)
    
    if not os.path.isdir("application_logs"):
        os.mkdir('application_logs')
    
    """FL Settings"""
    config.server_configuration["configuration"] = prefix
    config.server_configuration["learning_config"]["aggregator"] = args.aggregation
    config.server_configuration["learning_config"]["epochs"] = int(args.local_epochs)
    config.server_configuration["learning_config"]["K"] = int(args.number_of_clients)
    config.server_configuration["learning_config"]["rounds"] = int(args.number_of_training_rounds)
    
    """Dataset"""
    config.server_configuration["collection"]["selection"] = args.dataset
    config.server_configuration["collection"]["datasets"][args.dataset]["iid"] = iid
    config.server_configuration["collection"]["datasets"][args.dataset]["training_batch_size"] = int(args.training_batch_size)
    config.server_configuration["collection"]["datasets"][args.dataset]["testing_batch_size"] = int(args.testing_batch_size)
    config.server_configuration["collection"]["datasets"][args.dataset]["val_size"] = args.valid_size
    
    if not iid:
        config.server_configuration["collection"]["non_iid"]["type"] = args.partition
        config.server_configuration["collection"]["non_iid"]["alpha"] = float(args.dirichlet_alpha)
        
    """Models"""
    if args.model == "MLP": 
        model_name = f"{args.model}_{args.dataset}"
    else:
        model_name = args.model
    config.server_configuration["learning_config"]["global_model"]["type"] = model_name
    
    """Attack Configuration"""
    config.server_configuration["attack"]["engange"] = active_attack
    config.server_configuration["attack"]["engagement_criteria"]["epoch"] = int(args.engagement_criteria)
    config.server_configuration["attack"]["type"] = args.type_of_attack if args.type_of_optimisation == "bayesian_optimisation" else "single"
    config.server_configuration["attack"]["objective"] = args.objective
    config.server_configuration["attack"]["available_data"]["type"] = args.available_data
    config.server_configuration["attack"]["type_of_optimisation"] = args.type_of_optimisation
    config.server_configuration["attack"]["best_features"] = features_indices
    config.server_configuration["attack"]["num_of_neurons"] = int(args.neurons)
    config.server_configuration["attack"]["use_logits"] = use_logits
    config.server_configuration["attack"]["bounds"] = float(args.attack_bounds)
    config.server_configuration["learning_config"]["number_of_adversaries"] = int(args.number_of_adversaries)
    
    if args.objective == "multi":
        config.server_configuration["attack"]["multi_obj_bo"]["objectives"] = [args.objective_1st , args.objective_2nd]
    
    """Backend setup"""
    config.server_configuration["multiprocessing"] = mp
    config.server_configuration["device"] = args.device
    
    config.server_configuration["name"] = f"{prefix}_{args.aggregation}_{args.model}_{args.dataset}"
    
    print('Hello buddy! How are you today?', end='\n')
    print('We are about to start our experiment', end='\n')
    print(f"""This is the experiment we are going to run: 
Aggregation: {args.aggregation} 
Model: {args.model} 
Dataset: {args.dataset} 
Partition: {args.partition if not iid else 'IID'} 
Type of Optimisation: {args.type_of_optimisation}
Objective Function: {args.objective}""")
    
    seeds = [21325 for iter in range(1,(1+args.number_of_seeds))]#[2]#, 10, 42, 45, 48]

    if use_neurons:
        nums = [int(args.neurons)]
        
        if len(seeds) > 1:
            for num_of_neurons in nums:
                checkpoint_path = configure_checkpoint(config, checkpoint, seed, num_of_neurons, date_of_experiment=args.date_of_experiment)
            
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                run(seed, checkpoint, checkpoint_path, num_of_neurons)                
        else:
            for num_of_neurons, seed in zip(nums, seeds):
                checkpoint_path = configure_checkpoint(config, checkpoint, seed, num_of_neurons, date_of_experiment=args.date_of_experiment)
            
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                run(seed, checkpoint, checkpoint_path, num_of_neurons)
    
    else:
        num_of_neurons = 1
        
        for seed in seeds:
        
            checkpoint_path = configure_checkpoint(config, checkpoint, seed, num_of_neurons, date_of_experiment=args.date_of_experiment)    
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            run(seed, checkpoint, checkpoint_path, num_of_neurons)
        
def configure_checkpoint(config: dict, checkpoint:bool, seed:int, num_of_neurons:int, **kwargs) -> str:
    """ Check if the checkpoint path exists
        if checkpoint:
            identifies the path is correct eitherwise stops the execution
        else:
            removes any folder with that name
            Creates a new folder for the checkpoint

    Args:
        config (dict): server configuration
        checkpoint (bool): Start from a checkpoint True/False
        seed (int): The number of the seed
        num_of_neurons (int): the number of neurons the attack will apply

    Raises:
        FileExistsError: If there is no checkpoint assosiated with the current settings

    Returns:
        str: the path
    """
    d = datetime.datetime.now()
    date = kwargs["date_of_experiment"] if checkpoint else f"{d.year}{d.month if d.month>9 else f'0{d.month}'}{d.day if d.day>9 else f'0{d.day}'}"
    if config.server_configuration["attack"]["engage"] and args.number_of_adversaries !=0:
        checkpoint_path = f"{os.getcwd()}/checkpoint/{date}/{prefix}_{args.aggregation}_{args.model}_{args.dataset}_{args.partition if int(args.iid)==0 else 'iid'}_{config.server_configuration['attack']['type']}_{args.type_of_optimisation}_{config.server_configuration['attack']['objective']}_attackers_{args.number_of_adversaries}_neurons_{num_of_neurons}_seed_{seed}"
    else:
        checkpoint_path = f"{os.getcwd()}/checkpoint/{date}/{prefix}_{args.aggregation}_{args.model}_{args.dataset}_{args.partition if int(args.iid)==0 else 'iid'}_normal_seed_{seed}"    
    
    if checkpoint:
        if not os.path.isdir(checkpoint_path):
            raise FileExistsError("This experiment cannot be completed. There is no checkpoint")
    else:
        try:
            os.makedirs(checkpoint_path)
        except FileExistsError:
            shutil.rmtree(checkpoint_path)
            os.makedirs(checkpoint_path)
            
    return checkpoint_path
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model and Aggregation
    parser.add_argument('-a',       '--aggregation', help='The type of aggregation to perform.', default="pFedME")
    parser.add_argument('-m',       '--model', help='The model to use for the aggregation.', default="AlexNet")
    parser.add_argument('-le',      '--local_epochs', help='local epochs', default=3)
    parser.add_argument('-nc',      '--number_of_clients', help='Number of Clients', default=6)
    parser.add_argument('-na',      '--number_of_adversaries', help='Number of Adversaries', default=2)
    parser.add_argument('-tr',      '--number_of_training_rounds', help='Number of Training Rounds', default=51)
    
    # Dataset
    parser.add_argument('-d',       '--dataset', help='The dataset in use', default="CIFAR10")
    parser.add_argument('-p',       '--iid', help='iid=1 or non-iid=0', default=0)
    parser.add_argument('-tp',      '--partition', help='dirichlet / pathological / imbalanced', default="imbalanced")
    parser.add_argument('-da',      '--dirichlet_alpha', help='Set Alpha from 0 to 1', default=0.1)
    parser.add_argument('-ia',      '--imbalanced_ratio', help='Set Alpha from 0 to 1', default=0.25)
    parser.add_argument('-bstr',    '--training_batch_size', help='integer number for the training batch size', default=128)
    parser.add_argument('-bsts',    '--testing_batch_size', help='integer number for the testing batch size', default=256)
    parser.add_argument('-vs',      '--valid_size', help='Validation Size', default=0.15)
    
    # Attack Settings
    parser.add_argument('-aa',      '--active_attack', help='attack=1 / not attack=0', default=1) 
    parser.add_argument('-ec',      '--engagement_criteria', help='which epoch to engange', default=1) 
    parser.add_argument('-ta',      '--type_of_attack', help='single/multi', default="single")
    parser.add_argument('-ad',      '--available_data', help='min/max/random', default="max")
    parser.add_argument('-to',      '--type_of_optimisation', help='least_squares / bayesian_optimisation / rl', default="rl")
    parser.add_argument('-o',       '--objective', help='kl_div or multi_info or js_div', default="kl_div")
    parser.add_argument('-obj1',    '--objective_1st', help='kl_div or multi_info or js_div', default="kl_div")
    parser.add_argument('-obj2',    '--objective_2nd', help='kl_div or multi_info or js_div', default="multi_info")
    parser.add_argument('-n',       '--neurons', help='Number of Neurons', default=5)
    parser.add_argument('-fi',      '--features_indices', help='Indices of Neurons\n 0 - Continues \n 1 - Fixed ', default=0)
    parser.add_argument('-ul',      '--use_logits', help='Use Logits\n 0 - False \n 1 - True ', default=0)
    parser.add_argument('-ab',      '--attack_bounds', help='Use Logits\n 0 - False \n 1 - True ', default=1)
    
    # General Setup
    parser.add_argument('-ns',      '--number_of_seeds', help='specify the number of seeds', default=1)
    parser.add_argument('-de',      '--device', help='cpu or cuda', default="cpu")
    parser.add_argument('-mp',      '--multiprocessing', help='0=False, 1=True', default=0)
    parser.add_argument('-ch',      '--checkpoint', help='Start from check point\n 0=False, 1=True', default=0)
    parser.add_argument('-doe',     '--date_of_experiment', help='Specify the date in YYYYMMDD', default="0000000")
    
    args = parser.parse_args()
    
    if args.aggregation in ["FedAvg", "Krum", "FedProx"]:
        prefix = "FL"
    elif args.aggregation in ["FedDyn", "Ditto", "pFedME", "PerFedAvg"]:
        prefix = "PFL" 
    elif args.aggregation == "CML": 
        prefix = "CML"
    
    main(args, prefix)