from src.learning.server.serverBase import *
from src.learning.server.serverBaseMultiProcess import *
from collections import OrderedDict

import torch.nn as nn

class ServerSplitFed(ServerBase):
    def __init__(self, config, seed=0, checkpoint=False, checkpoint_path=""):
        super().__init__(config, seed, checkpoint, checkpoint_path)
        self.split_layer = self.server_config["learning_config"]["split_layer"]
        
        self.server_model, self.client_model = self.split_model()

        
        self.setup_client()
        self.setup_dataset()
        
        self.send_models_to_clients(round=0)

    def setup_client(self)  -> None:
        # Deploying the client models
        torch.cuda.empty_cache()
        
        from src.learning.client import ClientSplitFed as Client

        for i in range(0, self.num_workers):
            is_advesary = True if i in self.adversarial_clients else False
            self.list_of_clients.append(Client(i,self.device, adversary=is_advesary))
            print('\r Clients {}/{}'.format(i+1, self.num_workers), end='\r')

    def split_model(self, split_layer_index=0):
        """
        Split a PyTorch model into two parts at the specified layer index.
        
        Args:
        split_layer_index (int): The index of the layer at which to split the model.
        
        Returns:
        tuple: (client_model, server_model)
            - client_model (nn.Module): The part of the model before the split layer.
            - server_model (nn.Module): The part of the model from the split layer onwards.
        """
        split_layer_index = self.split_layer
        client_layers = OrderedDict()
        server_layers = OrderedDict()
        
        view_point, model_name = self.model.view_point, self.server_config["learning_config"]["global_model"]['type']
        
        non_layer_modules = (nn.ReLU, nn.MaxPool2d, nn.Dropout, 
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.LocalResponseNorm, nn.AdaptiveAvgPool2d)
        
        layer_counter = 0
        split_found = False

        def process_module(module, prefix=''):
            nonlocal layer_counter, split_found
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Sequential):
                    process_module(child, full_name)
                elif not isinstance(child, non_layer_modules):
                    if layer_counter == split_layer_index:
                        split_found = True
                    
                    if split_found:
                        server_layers[full_name.split(".")[-1]] = child
                    else:
                        client_layers[full_name.split(".")[-1]] = child
                    
                    layer_counter += 1
                else:
                    if not split_found:
                        client_layers[full_name.split(".")[-1]] = child
                    else:
                        server_layers[full_name.split(".")[-1]] = child

        process_module(self.model)
        
        if not split_found:
            raise ValueError(f"Split layer index {split_layer_index} is out of range. Model only has {layer_counter} layers.")
        
        class ClientModel(nn.Module):
            def __init__(self, view_point, model_name):
                super().__init__()
                self.layers = nn.ModuleDict(client_layers)
                self.model_name = model_name
                self.view_point = view_point
            
            def forward(self, x):
                for name, layer in self.layers.items():
                    x = layer(x)
                    x = x.view(x.size(0), self.view_point) if name == "avgpool" and self.model_name=="AlexNet" else x
                        
                return x
        
        class ServerModel(nn.Module):
            def __init__(self, view_point, model_name):
                super().__init__()
                self.layers = nn.ModuleDict(server_layers)
                self.model_name = model_name
                self.view_point = view_point
            
            def forward(self, x):
                for name, layer in self.layers.items():
                    x = layer(x)
                    x = x.view(x.size(0), self.view_point) if name == "avgpool" and self.model_name=="AlexNet" else x
                
                return x  
        
        return ServerModel(view_point, model_name), ClientModel(view_point, model_name)

    def fit(self):
        if self.checkpoint:
            self.load_from_checkpoint()
            training_round = int(np.loadtxt(f"{self.path}/epoch.txt"))
        else:
            training_round = 0

        for j in range(training_round, self.rounds):
            print(f"Round {j+1:3} / {self.rounds}")
            logger.info('==================\nStart of Round {:3}\n================\n'.format(j+1))
            
            client_updates = []
            for i in range(0, self.workers):
                client = self.list_of_clients[i]
                training_dataset = self.training_datasets[i]
                
                if client.trainable:
                    client_output = client.train(training_dataset, epoch=j)
                    client_updates.append(client_output)

            # Aggregate client updates and update server model
            self.update_global_model(client_updates)

            # Send updated models to clients
            self.send_models_to_clients(round=j+1)

            logger.info('GPU Temperature {}'.format(get_gpu_temperature()))
            logger.info('==================\Model Evaluation\n================\n')

            self.global_evaluation()
            self.personalised_evaluation()

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

    def update_global_model(self, client_updates):
        # Aggregate client updates (e.g., FedAvg)
        averaged_update = {k: torch.stack([update[k] for update in client_updates]).mean(0) 
                           for k in client_updates[0].keys()}
        
        # Update client-side of the global model
        self.client_model.load_state_dict(averaged_update)

    def send_models_to_clients(self, round):
        for client in self.list_of_clients:
            client.set_models(self.client_model, self.server_model)

    def process_client_activations(self, activations):
        # Forward pass through server model
        with torch.no_grad():
            output = self.server_model(activations)
        return output

    def get_full_model(self, model=None):
        full_model = copy.deepcopy(self.model)
        full_model.load_state_dict({**self.client_model.state_dict(), **self.server_model.state_dict()}) 
        return full_model