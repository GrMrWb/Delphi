from scipy.optimize import least_squares
from src.delphi.convex_opt.base_optimiser import *

class LeastSquareModelOptimiser(BaseOptimiser):
    def __init__(self, data, target, neural_net, config, feature_idx,target_confidence=0.25, **kwargs) -> None:
        super().__init__(data, target, neural_net, config, feature_idx,target_confidence=0.25, **kwargs)
        self.iteration = 0
        self.last_x = None
        self.obj_diff = None
        self.changes_of_feature = None
        
    def fun(self, x):
        y = self.model(x).numpy()
        self.iteration += 1
        self.last_x = x
        self.obj_diff = y - self.y_target
        print(self.obj_diff , end="\r")
        return self.obj_diff

    def jac(self):
        J = np.array([2])
        return J
    
    def run(self):        
        weights, _, _ = read_weights(self.neural_net, self.config)
        if len(weights.shape) > 2:
            initial_x0 = weights[self.feature_idx].flatten().cpu().detach().numpy()
        else:
            initial_x0 = torch.squeeze(weights[self.feature_idx].detach(), dim=0).cpu().numpy()
        res = least_squares(
            self.fun, 
            initial_x0,
            bounds=get_bounds(torch.squeeze(weights[self.feature_idx].view(1,-1), dim=0),self.config, numerical=True).numpy(), 
            ftol=0.1,
            )

        return res