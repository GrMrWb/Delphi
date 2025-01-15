from scipy.optimize import minimize
from src.delphi.convex_opt.base_optimiser import *

class L_BFGS_B(BaseOptimiser):
    def __init__(self, data, target, neural_net, config, feature_idx,target_confidence=0.25, **kwargs) -> None:
        super().__init__(data, target, neural_net, config, feature_idx,target_confidence=0.25, **kwargs)

    def jac(self):
        J = np.array([2])
        return J
    
    def run(self):
        def model(x):
            x = torch.tensor(x)
            
            net = copy.deepcopy(modify_the_weights_with_single_neuron(self.neural_net, x, self.device, self.feature_idx, server_config= self.config))
        
            self.data, self.y = self.data.to(self.device), self.y.to(self.device)
            net = net.to(self.device)
        
            output = net(self.data)
            
            output = F.softmax(output, dim=1)
            
            if self.target_confidence < .30:
                target_confidence = self.target_confidence*2
        
            if self.filtered_output:
                idx = 0
                for z, out in enumerate(output):
                    activate, activate_accuracy, activate_target_class, activate_confidence = False, False, False, False
                    if self.config["attack"]["filtering"]["type_of_filter"] in ("accuracy" or "all"):
                        activate_accuracy = torch.argmax(out) == self.y[z] and self.y[z] == self.target_class
                    if self.config["attack"]["filtering"]["type_of_filter"] in ("target_class" or "all"):
                        activate_target_class = self.y[z] == self.target_class
                    if self.config["attack"]["filtering"]["type_of_filter"] in ("confidence" or "all"):
                        activate_confidence = out.max() > target_confidence
                    
                    if self.config["attack"]["filtering"]["type_of_filter"] == "all":
                        activate = activate_accuracy and activate_confidence and activate_target_class
                
                    if activate_accuracy or activate_target_class or activate_confidence or activate:   
                        if idx == 0:
                            output_filtered = out
                            output_filtered = output_filtered.reshape(1, out.shape[0])
                            target_filtered = self.y[z].reshape(1,1)
                            idx+=1
                        else:
                            output_filtered = torch.cat([out.reshape(1, out.shape[0]), output_filtered])
                            target_filtered = torch.cat([self.y[z].reshape(1,1), target_filtered])
                            
                        if output_filtered.shape[0] > 50:
                            break
                if "output_filtered" not in locals():
                    output_filtered = output
                    target_filtered = self.y
            else:
                output_filtered = output
                target_filtered = self.y
        
            target = confidence_level(target_filtered, target_confidence=self.target_confidence, classes=output.shape[1])

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

        def fun(x, *args):
            y = model(x).numpy()
            print(y, end="\r")
            return y - self.y_target
        
        weights, _, _ = read_weights(self.neural_net, self.config)
        
        array = get_bounds(torch.squeeze(weights[self.feature_idx].view(1,-1), dim=0),self.config).numpy()
        
        bounds = np.empty((array.shape[1], array.shape[0]), dtype=array.dtype)
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                bounds[col, row] = array[row, col]
        
        res = minimize(
            fun, 
            torch.squeeze(weights[self.feature_idx].detach(), dim=0).cpu().numpy(),
            method = 'L-BFGS-B',
            bounds=bounds,
            )

        return res