from src.delphi.utils import *

class BaseOptimiser:
    def __init__(self, data, target, neural_net, config, feature_idx,target_confidence=0.25, **kwargs) -> None:
        self.data = data
        self.y = target
        self.neural_net = neural_net
        self.config = config
        self.device = config["device"]
        self.score= []
        self.feature_idx = feature_idx
        self.target_confidence = target_confidence
        self.optimisation_objective = None
        
        self.target_class = self.config["attack"]["target_class"]
        self.filtered_output = self.config["attack"]["filtering"]["filtered_output"]
        
        if "kl_div" in kwargs:
            self.optimisation_objective = "kl_div" if kwargs["kl_div"] else self.optimisation_objective
            self.y_target = self.config["attack"]["objectives"]["kl_div_target"]
        elif "mutual_info" in kwargs:
            self.optimisation_objective = "mutual_info" if kwargs["mutual_info"] else self.optimisation_objective
            self.y_target = self.config["attack"]["objectives"]["mutual_info_target"]
        elif "wasserstein" in kwargs:
            self.optimisation_objective = "wasserstein" if kwargs["wasserstein"] else self.optimisation_objective
            self.y_target = self.config["attack"]["objectives"]["wasserstein_target"]
        
        print(self.optimisation_objective)
    
    def model(self,x):
        x = torch.tensor(x, dtype=torch.float32)
        
        weights, _, _ = read_weights(self.neural_net, self.config)
        
        if len(weights.shape)>2:
            x = x.view(weights.shape[1], weights.shape[2], weights.shape[3])
        
        net = copy.deepcopy(modify_the_weights_with_single_neuron(self.neural_net, x, self.device, self.feature_idx, server_config= self.config))
       
        self.data, self.y = self.data.to(self.device), self.y.to(self.device)
        net = net.to(self.device)
       
        output = net(self.data)
        
        output = F.softmax(output, dim=1)
        
        if self.target_confidence < .30:
            target_confidence = self.target_confidence*2
            
        target_class = torch.tensor(self.config["attack"]["target_class"], device=self.device)
    
        if self.filtered_output:
            idx = 0
            for j in range(len(output)):
                if torch.argmax(output[j]) == self.y[j] and self.y[j] in target_class:
                    if idx == 0:
                        output_filtered = output[j]
                        output_filtered = output_filtered.reshape(1, output_filtered.shape[0])
                    else:
                        output_filtered = torch.cat((output_filtered, output[j].reshape(1,output[j].shape[0])))
                    
                    idx += 1

                if idx > 50: 
                    break
                
            output_filtered = torch.tensor(output_filtered)
            if "output_filtered" not in locals():
                output_filtered = output
        else:
            output_filtered = output

        if len(output_filtered.shape) < 1:
            output_filtered = output_filtered.reshape(1, output_filtered.shape[0]) 
       
        target = confidence_level(output_filtered, target_confidence=self.target_confidence, classes=output.shape[1])

        if self.optimisation_objective == "kl_div":
            kl_div_score = torch.Tensor([kl_divergence(output_filtered, target)])
            if torch.isnan(kl_div_score):
                score = torch.Tensor([10]) 
            else:
                score = kl_div_score
        
        elif self.optimisation_objective == "mutual_info":
            mi_score = torch.Tensor([mi_BO( output_filtered, target)])
            score = mi_score
            
        elif self.optimisation_objective == "wasserstein":
            wassertain_score = torch.Tensor([wasserstein_matrix(output_filtered, target)])
            score = wassertain_score
        
        self.score.append(score)    
        
        return score