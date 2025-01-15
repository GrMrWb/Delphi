from src.learning.client.clientBase import *

class ClientFedDyn(ClientBase):
    def __init__(self, client_id, device, **kwargs):
        super().__init__(client_id, device, **kwargs)
        
        self.global_model = None
                
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        self.alpha = self.server_config[f"{self.server_config['configuration']}_config"]["aggregation"][type(self).__name__.split('Client')[1]]["alpha"]

    def set_parameters(self, model):
        # parameters = model if isinstance(model, list) else model.parameters()
        
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            
        # self.model = copy.deepcopy(model)

        self.global_model = model_parameter_vector(model).detach().clone()
        
    def __client_normal__(self, dataset: object, **kwargs) -> list:
        self.model.train()
        
        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])

        torch.cuda.empty_cache()
        
        old_model = copy.deepcopy(self.model)
        self.model= self.model.to(self.device) 
        
        dataload = dataset['dataloader']
        for epoch in range(0,self.epochs):
            for batch_idx, (data, target) in enumerate(dataload):
                
                loss_func = self.__train__(data, target,batch_idx, size, epoch)
            
        print(f'\rClient {self.client_id:02} | Loss: {loss_func.item():.5f} | Batch: {size:03}/{size:03} | Epoch: {epoch+1:02} ', end='\n')
        
        if self.global_model != None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model)
        
        self.training_performance.append(loss_func.item())
        
        self.model = self.model.to('cpu')
        
        print("\rCalculating gradients", end="\r")
        self.gradients.append(get_historical_gradients(self.model, dataset["dataloader"], self.server_config))
        self.gradients.append(get_historical_gradients(old_model, dataset["dataloader"], self.server_config))
        
        print("\rCalculating Distribution", end="\r")
        # mean, std = get_historical_distribution(self.model, self.server_config)
        weights, _, _ = read_weights(self.model, self.server_config)
        
        self.distributions_mean.append(weights)
            
        del old_model
        del dataset, loss_func
        
        torch.cuda.empty_cache()

    def __train__(self, data: object, target: object, batch_idx: int, size: int, epoch:int, **kwargs):
        data, target = data.to(self.device), target.to(self.device)
        self.model = self.model.to(self.device)
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
               
        output = self.model(data)
        loss_func = self.loss(output, target)
        
        if self.global_model != None:
            v1 = model_parameter_vector(self.model)
            
            self.global_model = self.global_model.to(v1.device)
            self.old_grad = self.old_grad.to(v1.device)
            
            loss_func += (self.alpha/2) * torch.norm(v1 - self.global_model, 2)
            loss_func -= torch.dot(v1, self.old_grad)
        
        loss_func.backward()
        
        self.optimizer.step()
        
        print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Epoch: {:02} '.format(self.client_id,loss_func.item(),batch_idx+1, size, epoch+1), end='\r')
        
        del target, data, output
        torch.cuda.empty_cache()
        
        return loss_func
    
def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)