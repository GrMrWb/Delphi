import torch
import torch.nn as nn
from src.learning.client.clientBase import *

class ClientSplitFed(ClientBase):
    def __init__(self, client_id, device, **kwargs):
        super().__init__(client_id, device, **kwargs)
        self.split_layer = self.server_config["learning_config"]["split_layer"]
        self.client_model = None
        self.server_model = None
        

    def set_models(self, client_model, server_model, round=0):
        self.client_model = client_model
        if round==1:
            self.optimizer = choose_optimizer(self.client_model.parameters(), self.client_config["optimizer"])
        
        self.server_model = server_model

    def __client_normal__(self, dataset: object, **kwargs) -> list:
        torch._dynamo.config.suppress_errors = True
        self.client_model.train()
        self.server_model.eval()  # Server model is used for forward pass only

        torch.cuda.empty_cache()

        old_client_model = copy.deepcopy(self.client_model)
        self.client_model = self.client_model.to(self.device)
        self.server_model = self.server_model.to(self.device)

        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])

        for epoch in range(0, self.epochs):
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                loss_func = self.__train_split__(data, target, batch_idx, size, epoch)

            print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Local Epoch: {:02} '.format(self.client_id, loss_func.item(), batch_idx+1, size, epoch+1), end='\r')

        print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Local Epoch: {:02} '.format(self.client_id, loss_func.item(), batch_idx+1, size, epoch+1), end='\n')

        self.training_performance.append(loss_func.item())

        self.client_model = self.client_model.to('cpu')
        self.server_model = self.server_model.to('cpu')

        print("\rCalculating gradients", end="\r")
        self.gradients.append(get_historical_gradients(self.get_full_model(), dataset["dataloader"], self.server_config))
        self.gradients.append(get_historical_gradients(self.get_full_model(old_client_model), dataset["dataloader"], self.server_config))

        print("\rCalculating Distribution", end="\r")
        # mean, std = get_historical_distribution(self.model, self.server_config)
        weights, _, _ = read_weights(self.client_model, self.server_config)
        
        self.distributions_mean.append(weights)
        del old_client_model
        del dataset, loss_func

        torch.cuda.empty_cache()

        return self.client_model.state_dict()  # Return client model updates

    def __train_split__(self, data: object, target: object, batch_idx: int, size: int, epoch: int):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()

        # Forward pass through client model
        client_output = self.client_model(data)
        
        input_for_server = client_output
        
        input_for_server = input_for_server.to(self.device)
        input_for_server.retain_grad()
        
        # Forward pass through server model
        server_output = self.server_model(input_for_server)

        # Compute loss
        loss = self.loss(server_output, target)

        # Backward pass
        loss.backward(retain_graph=True)
        
        client_output.backward(input_for_server.grad)

        # Update client model
        self.optimizer.step()

        print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03}'.format(self.client_id, loss.item(), batch_idx, size), end='\r')

        del target, data, client_output, server_output

        torch.cuda.empty_cache()

        return loss

    def get_parameters(self):
        return self.client_model.state_dict()

    def get_full_model(self, model=None):
        full_model = nn.Sequential()
        full_model.add_module('client_model', self.client_model) if model == None else full_model.add_module('client_model', model)
        full_model.add_module('server_model', self.server_model)
        return full_model