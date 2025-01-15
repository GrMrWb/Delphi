import torch
import torch.nn as nn
import torch.nn.functional as F

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10, **kwargs):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class elu(nn.Module):
    def __init__(self) -> None:
        super(elu, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, x, 0.2 * (torch.exp(x) - 1))


class linear(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(linear, self).__init__()
        self.w = nn.Parameter(
            torch.randn(out_c, in_c) * torch.sqrt(torch.tensor(2 / in_c))
        )
        self.b = nn.Parameter(torch.randn(out_c))

    def forward(self, x):
        return F.linear(x, self.w, self.b)

class MLP_MNIST(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 80),
            nn.ELU(),
            nn.Linear(80, 60),
            nn.ELU(),
            nn.Linear(60, 10),
            nn.ELU(),
        )
    
    def forward(self, x):
        x.to(dtype=torch.float64)
        return self.network(x)

# class MLP_MNIST(nn.Module):
#     def __init__(self, **kwargs) -> None:
#         super(MLP_MNIST, self).__init__()
#         self.fc1 = linear(28 * 28, 80)
#         self.fc2 = linear(80, 60)
#         self.fc3 = linear(60, 10)
#         self.flatten = nn.Flatten()
#         self.activation = elu()

#     def forward(self, x):
#         x = self.flatten(x)

#         x = self.fc1(x)
#         x = self.activation(x)

#         x = self.fc2(x)
#         x = self.activation(x)

#         x = self.fc3(x)
#         x = self.activation(x)

#         return x
    
class MLP_CIFAR10(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 80)
        self.fc2 = nn.Linear(80, 60)
        self.fc3 = nn.Linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        # x = self.activation(x)

        return x
    
class MLP_CIFAR100(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(MLP_CIFAR100, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 80)
        self.fc2 = nn.Linear(80, 60)
        self.fc3 = nn.Linear(60, 100)
        self.flatten = nn.Flatten()
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        # x = self.activation(x)

        return x