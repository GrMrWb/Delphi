from src.learning.client.clientBase import *

class ClientPerFedAvg(ClientBase):
    def __init__(self, client_id, device, hessian_free=False, **kwargs):
        super().__init__(client_id, device, **kwargs)
        
        self.global_model = copy.deepcopy(self.model)
        
        self.alpha = 1e-2
        self.beta = 1e-3
        self.hessian_free = hessian_free
            
    def get_data_batch(self, dataset):
        try:
            x, y = next(dataset)
        except:
            dataset = iter(dataset)
            x, y = next(dataset)

        return x.to(self.device), y.to(self.device)

    @torch.compile(mode="reduce-overhead", disable=True) 
    def __client_normal__(self, dataset: object, **kwargs) -> list:
        self.model.train()
        old_model = copy.deepcopy(self.model)
        self.model = self.model.to(self.device)

        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])

        torch.cuda.empty_cache()

        if self.hessian_free:  # Per-FedAvg(HF)
            for epoch in range(self.epochs):
                for batch_idx in range(0, size):
                    temp_model = copy.deepcopy(self.model)
                    data_batch_1 = self.get_data_batch(dataset['dataloader'])
                    grads = self.compute_grad(temp_model, data_batch_1)
                    for param, grad in zip(temp_model.parameters(), grads):
                        param.data.sub_(self.alpha * grad)

                    data_batch_2 = self.get_data_batch(dataset['dataloader'])
                    grads_1st = self.compute_grad(temp_model, data_batch_2)

                    data_batch_3 = self.get_data_batch(dataset['dataloader'])

                    grads_2nd = self.compute_grad(self.model, data_batch_3, v=grads_1st, second_order_grads=True)
                    
                    for param, grad1, grad2 in zip(self.model.parameters(), grads_1st, grads_2nd):
                        param.data.sub_(self.beta * grad1 - self.beta * self.alpha * grad2)

                    data,target = data_batch_1
                    output = self.model(data)
                    loss_func = self.loss(output, target)
                    
                    print(f'\rClient {self.client_id:02} | Loss: {loss_func.item():.5f} | Batch: {batch_idx:03} | Epoch: {epoch+1:02}', end='\r')  
                
        else:  # Per-FedAvg(FO)
            for epoch in range(self.epochs):
                for batch_idx in range(0, size):
                    temp_model = copy.deepcopy(self.model)
                    data_batch_1 = self.get_data_batch(dataset['dataloader'])
                    grads = self.compute_grad(temp_model, data_batch_1)

                    for param, grad in zip(temp_model.parameters(), grads):
                        param.data.sub_(self.alpha * grad)

                    data_batch_2 = self.get_data_batch(dataset['dataloader'])
                    grads = self.compute_grad(temp_model, data_batch_2)

                    for param, grad in zip(self.model.parameters(), grads):
                        param.data.sub_(self.beta * grad)
            
                    data,target = data_batch_1
                    output = self.model(data)
                    loss_func = self.loss(output, target)
                    
                    print(f'\rClient {self.client_id:02} | Loss: {loss_func.item():.5f} | Batch: {batch_idx:03} | Epoch: {epoch+1:02}', end='\r')  
                
        print(f'\rClient {self.client_id:02} | Loss: {loss_func.item():.5f} | Batch: {batch_idx:03} | Epoch: {epoch+1:02}', end='\n')  
        
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
        
        self.global_model = copy.deepcopy(self.model)
      
    @torch.compile(mode="reduce-overhead", disable=True) 
    def compute_grad(
        self,
        model: torch.nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        v: Union[Tuple[torch.Tensor, ...], None] = None,
        second_order_grads=False,
    ):
        x, y = data_batch
        if second_order_grads:
            frz_model_params = copy.deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = self.model(x)
            loss_1 = self.loss(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, self.model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = self.model(x)
            loss_2 = self.loss(logit_2, y)
            grads_2 = torch.autograd.grad(loss_2, self.model.parameters())

            model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads

        else:
            logit = model(x)
            loss = self.loss(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads