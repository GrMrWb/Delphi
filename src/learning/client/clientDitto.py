from src.learning.client.clientBase import *
from src.learning.utils import PerturbedGradientDescent

class ClientDitto(ClientBase):
    def __init__(self, client_id, device, **kwargs):
        super().__init__(client_id, device, **kwargs)
        
        self.personalised_model = copy.deepcopy(self.model)
        
        self.mu = self.server_config["learning_config"]["aggregation"][self.server_config["learning_config"]["aggregator"]]["mu"]
        self.plocal_steps = self.server_config["learning_config"]["aggregation"][self.server_config["learning_config"]["aggregator"]]["plocal_steps"]
        self.learning_rate = 0.001
        
        self.optimizer_per = PerturbedGradientDescent(self.personalised_model.parameters(), lr=self.learning_rate, mu=self.mu)
        
        self.per_validation_performance = [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric", "confidence", 
                "mi", "correctness", "entropy", "variability", "loss", "ece", "mce"
            ],
        ]
        self.per_testing_performance = [
            [
                "client_id", "accuracy", "precision","recall", "aleatoric", "confidence",
                 "mi", "correctness", "entropy", "variability", "loss", "ece", "mce"
            ],
        ]

    
    @torch.compile(mode="reduce-overhead", disable=True)
    def continue_training(self, dataset, epochs, **kwargs):
        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])

        start = 0 if epochs==1 else 1
        end = epochs
        for epoch in range(start,end):
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                data, target = data.to(self.device), target.to(self.device)

                loss_func = self.__train__(data, target,batch_idx, size, epoch) if "global_train" in kwargs else self.__personalised_train__(data, target,batch_idx, size, epoch)
            
            print(f'\rClient {self.client_id:02} | {"Global" if "global_train" in kwargs else "Personalised"} | Loss: {loss_func.item():.5f} | Batch: { batch_idx+1:03}/{size:03} | Epoch: { epoch+1:02} ', end='\r')
        
        try:
            print(f'\rClient {self.client_id:02} | {"Global" if "global_train" in kwargs else "Personalised"} | Loss: {loss_func.item():.5f} | Batch: { batch_idx+1:03}/{size:03} | Epoch: { epoch+1:02} ', end='\n')
        except:
            pass
            
        return loss_func
    
    @torch.compile(mode="reduce-overhead", disable=True)
    def __client_normal__(self, dataset: object, **kwargs) -> list:
        self.model.train()
        self.personalised_model.train()
        
        size = int(len(dataset['dataloader'].dataset)/self.server_config["collection"]["datasets"][self.server_config["collection"]["selection"]]["training_batch_size"])

        torch.cuda.empty_cache()
        old_model = copy.deepcopy(self.model)
        self.model = self.model.to(self.device)
        self.personalised_model = self.personalised_model.to(self.device)
        
        dataload = dataset['dataloader']
        for epoch in range(0, self.plocal_steps):
            for batch_idx, (data, target) in enumerate(dataload):
                data, target = data.to(self.device), target.to(self.device)
                loss_func = self.__personalised_train__(data, target,batch_idx, size, epoch)
        
        print(f'\rClient {self.client_id:02} | Personalised | Loss: {loss_func.item():.5f} | Batch: {size:03}/{size:03} | Epoch: {epoch+1:02} ', end='\r')
        
        self.personalised_model = self.personalised_model.to('cpu')
        
        print("\r                                                                                                                              ", end='\r')
        for epoch in range(0,self.epochs):
            for batch_idx, (data, target) in enumerate(dataload):
                data, target = data.to(self.device), target.to(self.device)
                loss_func = self.__train__(data, target,batch_idx, size, epoch)
            
        print(f'\rClient {self.client_id:02} | Global | Loss: {loss_func.item():.5f} | Batch: {size:03}/{size:03} | Epoch: {epoch+1:02} ', end='\n')

        self.training_performance.append(loss_func.item())
        
        self.model = self.model.to('cpu')
        
        # print("\rCalculating gradients", end="\r")
        # self.gradients.append(get_historical_gradients(self.model, dataset["dataloader"], self.server_config))
        # self.gradients.append(get_historical_gradients(old_model, dataset["dataloader"], self.server_config))
        
        print("\rCalculating Distribution", end="\r")
        # mean, std = get_historical_distribution(self.model, self.server_config)
        weights, _, _ = read_weights(self.model, self.server_config)
        
        self.distributions_mean.append(weights)
            
        del old_model
        del dataset, loss_func
        
        torch.cuda.empty_cache()
    
    @torch.compile(mode="reduce-overhead", disable=True)  
    def __train__(self, data: object, target: object, batch_idx: int, size: int, epoch:int, **kwargs):
        torch.cuda.empty_cache()
        self.optimizer.zero_grad() 
               
        output = self.model(data)

        loss_func = self.loss(output, target)
        
        loss_func.backward()
        
        self.optimizer.step()
        
        print(f"\rClient {self.client_id:02} | Global | Loss: {loss_func.item():.5f} | Batch: {batch_idx+1:03}/{size:03} | Epoch: {epoch+1:02} ", end='\r')
        
        # del target, data, output
        # self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        
        return loss_func
    
    @torch.compile(mode="reduce-overhead", disable=True) 
    def __personalised_train__(self, data: object, target: object, batch_idx: int, size: int, epoch:int, **kwargs):
        torch._dynamo.config.suppress_errors = True
        self.optimizer_per.zero_grad()
        
        torch.cuda.empty_cache()
        
        output = self.personalised_model(data)
        
        loss_func = self.loss(output, target)
        loss_func.backward()
        
        self.optimizer_per.step(self.model.parameters(), self.device)
        
        print(f"\rClient {self.client_id:02} | Personalised | Loss: {loss_func.item():.5f} | Batch: {batch_idx+1:03}/{size:03} | Epoch: {epoch+1:02} ", end='\r')
        
        del target, data, output
        torch.cuda.empty_cache()
        
        return loss_func
    
    def per_validation(self, dataset) -> None:
        accuracy, precision, recall, aleatoric, confidence = 0, 0, 0, 0, 0
        mi, correctness, entropy, variability,loss = 0, 0, 0, 0, 0
        
        self.personalised_model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataset['dataloader']):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    self.personalised_model= self.personalised_model.to(self.device)
                    output = self.personalised_model(data)
                    
                    loss_func = self.loss(output, target)
                    
                    probabilities = F.softmax(output, dim=1)
                    
                    loss += loss_func.item()
                    
                    if batch_idx==0:
                        prob_array = probabilities
                        target_array = target
                    else:
                        prob_array = torch.cat([prob_array, probabilities]) 
                        target_array = torch.cat([target_array, target]) 
                except:
                    pass
                    
            
            results = evaluate_predictions(target_array, prob_array)
                
        accuracy = results["accuracy"]
        precision = results["precision"]
        recall = results["recall"]
        aleatoric = results["aleatoric"]
        confidence = results["confidence"]
        mi = results["mi"]
        correctness = results["correctness"]
        entropy = results["entropy"]
        variability = results["variability"]
        
        loss /= (batch_idx + 1)
        
        self.per_validation_performance.append([
            f'Client_id: {self.client_id}',
            accuracy,
            precision,
            recall,
            aleatoric,
            confidence,
            mi,
            correctness,
            entropy,
            variability,
            loss,
            results["ece"],
            results["mce"],
        ])
        
        print(f"\rClient {self.client_id:02} | Accuracy: {accuracy:.5f} | MI:  {mi:.5f} | Correctness: {correctness:.5f}", end="\r")
                
    def per_testing(self, dataset) -> None:
        accuracy, precision, recall, aleatoric, confidence = 0, 0, 0, 0, 0
        mi, correctness, entropy, variability,loss = 0, 0, 0, 0, 0
        
        self.personalised_model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataset):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    self.personalised_model= self.personalised_model.to(self.device)
                    output = self.personalised_model(data)
                    
                    loss_func = self.loss(output, target)
                    
                    probabilities = F.softmax(output, dim=1)
                    loss += loss_func.item()
                    
                    if batch_idx==0:
                        prob_array = probabilities
                        target_array = target
                    else:
                        prob_array = torch.cat([prob_array, probabilities]) 
                        target_array = torch.cat([target_array, target]) 
                except:
                    pass
            
            results = evaluate_predictions(target_array, prob_array)
                
        accuracy = results["accuracy"]
        precision = results["precision"]
        recall = results["recall"]
        aleatoric = results["aleatoric"]
        confidence = results["confidence"]
        mi = results["mi"]
        correctness = results["correctness"]
        entropy = results["entropy"]
        variability = results["variability"]
        
        loss /= (batch_idx + 1)
        
        self.per_testing_performance.append([
            f'Client_id: {self.client_id}',
            accuracy,
            precision,
            recall,
            aleatoric,
            confidence,
            mi,
            correctness,
            entropy,
            variability,
            loss,
            results["ece"],
            results["mce"],
        ])
        
        print(f"\rClient {self.client_id:02} | Accuracy: {accuracy:.5f} | MI:  {mi:.5f} | Correctness: {correctness:.5f}", end="\r")        