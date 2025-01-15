from src.learning.server.serverBase import *
from src.learning.server.serverBaseMultiProcess import *
from random import shuffle

import config

configuration = config.server_configuration

if configuration["multiprocessing"]:
    BaseClass = ServerBaseMultiProcess
else:
    BaseClass = ServerBase

class ServerFedDyn(BaseClass):
    def __init__(self, config, seed=0, checkpoint=False) -> None:
        super().__init__(config, seed, checkpoint)
        
        self.personalised_loss = []
        self.generalised_loss = []
        self.personalised_accuracy = []
        self.generalised_accuracy = []
        
        self.alpha = self.server_config["learning_config"]["aggregation"][type(self).__name__.split('Server')[1]]["alpha"]
        self.server_state = copy.deepcopy(self.model)
        
        self.setup_client()
        self.setup_dataset()
        
    def setup_client(self)  -> None:
        # Deploying the client models
        torch.cuda.empty_cache()
        
        if self.aggregation == "FedDyn":
            from src.learning.client import ClientFedDyn as Client
        else:
            raise NotImplementedError 
        
        for i in range(0, self.num_workers):
            is_advesary = True if i in self.adversarial_clients else False
            self.list_of_clients.append(Client(i,self.device, adversary=is_advesary))
            print('\r Clients {}/{}'.format(i+1, self.num_workers), end='\r')
            
    def add_parameters(self, client_model):
        for server_param, client_param in zip(self.model.parameters(), client_model.model.parameters()):
            server_param.data += client_param.data.clone() / self.workers

    def aggregate_parameters(self):
        weights = copy.deepcopy(self.list_of_clients)
        shuffle(weights)
        self.model = copy.deepcopy(weights[0].model)
        
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for client_model in self.list_of_clients:
            self.add_parameters(client_model)

        for server_param, state_param in zip(self.model.parameters(), self.server_state.parameters()):
            server_param.data -= (1/self.alpha) * state_param

    def update_server_state(self):
        weights = copy.deepcopy(self.list_of_clients)
        shuffle(weights)
        model_delta= copy.deepcopy(weights[0].model)
        
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.list_of_clients:
            for server_param, client_param, delta_param in zip(self.model.parameters(), client_model.model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.workers

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param