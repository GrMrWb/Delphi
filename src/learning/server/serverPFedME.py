from src.learning.server.serverBase import *
from src.learning.server.serverBaseMultiProcess import *

import config
import pickle

configuration = config.server_configuration

if configuration["multiprocessing"]:
    BaseClass = ServerBaseMultiProcess
else:
    BaseClass = ServerBase

class ServerPFedME(BaseClass):
    def __init__(self, config, seed=0, checkpoint=False, checkpoint_path="") -> None:
        super().__init__(config, seed, checkpoint, checkpoint_path)
        
        self.personalised_loss = []
        self.generalised_loss = []
        self.personalised_accuracy = []
        self.generalised_accuracy = []
        
        self.setup_client()
        self.setup_dataset()
        
    def setup_client(self)  -> None:
        # Deploying the client models
        torch.cuda.empty_cache()
        
        if self.aggregation == "pFedME":
            from src.learning.client import ClientPFedME as Client
        else:
            raise NotImplementedError 
        
        for i in range(0, self.num_workers):
            is_advesary = True if i in self.adversarial_clients else False
            self.list_of_clients.append(Client(i,self.device, adversary=is_advesary))
            print('\r Clients {}/{}'.format(i+1, self.num_workers), end='\r')