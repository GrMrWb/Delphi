from src.learning.client.clientBase import *

class ClientFedProx(ClientBase):
    def __init__(self, client_id, device, fedproxloss_mu=0.1, **kwargs):
        super().__init__(client_id, device, **kwargs)
        
        self.global_model = copy.deepcopy(self.model)
        
        self.fedproxloss_mu = self.server_config["learning_config"]["aggregation"][type(self).__name__.split('Client')[1]]["mu"]
        self.loss_prox = PTFedProxLoss(mu=self.fedproxloss_mu)
        
    def set_parameters(self, model):
        parameters = model if isinstance(model, list) else model.parameters()
        
        for old_param_global_model, old_param_local_model, new_param in zip(self.model.parameters(), self.global_model.parameters(), parameters):
            old_param_global_model.data = new_param.data.clone()
            old_param_local_model.data = new_param.data.clone()
    
    @torch.compile(mode="reduce-overhead", disable=True) 
    def __client_normal__(self, dataset: object, **kwargs) -> list:
        self.model.train()
        # data_iq = DataIQ_Torch(X=dataset['full_dataset'][0], y=dataset['full_dataset'][1])

        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])

        torch.cuda.empty_cache()
        
        self.model= self.model.to(self.device) 
        old_model = copy.deepcopy(self.model)
        
        activate_fed_prox = True if kwargs["epoch"] > 0 else False
        dataload = dataset['dataloader']
        for epoch in range(0,self.epochs):
            for batch_idx, (data, target) in enumerate(dataload):
                
                loss_func = self.__train__(data, target,batch_idx, size, epoch, activate=activate_fed_prox)
            
        # print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} at {:03} '.format(self.client_id, loss_func.item(), size, size, epoch+1), end='\n')
            
        print(f'\rClient {self.client_id:02} | Loss: {loss_func.item():.5f} | Batch: {size:03}/{size:03} | Epoch: {epoch+1:02} ', end='\n')
        
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
    
    @torch.compile(mode="reduce-overhead", disable=True) 
    def __train__(self, data: object, target: object, batch_idx: int, size: int, epoch:int, **kwargs):
        data, target = data.to(self.device), target.to(self.device)
        self.model= self.model.to(self.device)
        self.optimizer.zero_grad() 
        torch.cuda.empty_cache()
               
        output = self.model(data)

        loss_func = self.loss(output, target)
        
        if "activate" in kwargs:
            
            loss_func_prox = self.loss_prox(self.model, self.global_model.to(self.device)) if kwargs["activate"] else 0
            
            loss_func += loss_func_prox
        
        loss_func.backward()
        
        self.optimizer.step()
        
        print('\rClient {:02} | Loss: {:.5f} | Batch: {:03}/{:03} | Epoch: {:02} '.format(self.client_id,loss_func.item(),batch_idx+1, size, epoch+1), end='\r')
        
        # del target, data, output
        # self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        
        return loss_func