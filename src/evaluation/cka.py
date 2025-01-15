import math
import torch
from torch_cka import CKA

def calculate_cka_model(new_model, old_model, dataset, device):
    for name, param in new_model.named_parameters():
        break
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        cka = CKA(
            new_model, 
            old_model,
            model1_name="NewModel", 
            model2_name="OldModel",
            device=device,
            model1_layers=[name],
            model2_layers=[name],
        )
        
        cka.compare(dataset)
        
    info = cka.export()["CKA"]
    
    del cka, new_model, old_model, dataset
    
    return info

def calculate_cka_layer(new_layer, old_layer, device):
    cka = CudaCKA(device)
    score = cka.linear_CKA(new_layer, old_layer)
    
    return score

class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)