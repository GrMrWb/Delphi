from src.learning.models.AlexNet import AlexNet
from src.learning.models.ExplainableAlexNet import ExplainableAlexNet
from src.learning.models.test_model import Mclr_CrossEntropy, MLP_MNIST, MLP_CIFAR10, MLP_CIFAR100
from torchvision.models.resnet import resnet18 
from torchvision.models.resnet import resnet34 
from torchvision.models.resnet import resnet50 
from torchvision.models.resnet import resnet101
from torchvision.models.resnet import resnet152
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19
from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.swin_transformer import swin_t
from torchvision.models.vision_transformer import vit_b_16 as VisionTransformer

import torch

import config

configuration = config.server_configuration
prefix = "{}_".format(configuration["configuration"])
aggregation = configuration["learning_config"]['aggregator']

dataset = configuration['collection']['selection']

Resnet18 = resnet18()
Resnet18.fc = torch.nn.Linear(512, configuration['collection']["datasets"][dataset]["classes"], bias=True)

Resnet34 = resnet34()
Resnet34.fc = torch.nn.Linear(512, configuration['collection']["datasets"][dataset]["classes"], bias=True)

Resnet50 = resnet50()
Resnet50.fc = torch.nn.Linear(2048, configuration['collection']["datasets"][dataset]["classes"], bias=True)

Resnet101 = resnet101()
Resnet101.fc = torch.nn.Linear(2048, configuration['collection']["datasets"][dataset]["classes"], bias=True)

Resnet152 = resnet152()
Resnet152.fc = torch.nn.Linear(512, configuration['collection']["datasets"][dataset]["classes"], bias=True)

VGG11 = vgg11()
VGG11.classifier[6] = torch.nn.Linear(4096, configuration['collection']["datasets"][dataset]["classes"], bias=True)

VGG13 = vgg13()
VGG13.classifier[6] = torch.nn.Linear(4096, configuration['collection']["datasets"][dataset]["classes"], bias=True)

VGG16 = vgg16()
VGG16.classifier[6] = torch.nn.Linear(4096, configuration['collection']["datasets"][dataset]["classes"], bias=True)

VGG19 = vgg19()
VGG19.classifier[6] = torch.nn.Linear(4096, configuration['collection']["datasets"][dataset]["classes"], bias=True)

MobileNet = mobilenet_v3_large()
MobileNet.classifier[3] = torch.nn.Linear(1280, configuration["collection"]["datasets"][dataset]["classes"], bias=True)

SwinTransformer = swin_t()
SwinTransformer.head = torch.nn.Linear(768, configuration['collection']["datasets"][dataset]["classes"], bias=True)

AlexNet = AlexNet(channels = configuration['collection']["datasets"][dataset]["channels"], num_classes=configuration['collection']["datasets"][dataset]["classes"])
AlexNet.classifier.fc_layer_3 = torch.nn.Linear(4096, configuration['collection']["datasets"][dataset]["classes"], bias=True)

MLP_CIFAR10 = MLP_CIFAR10()
MLP_CIFAR100 = MLP_CIFAR100()

__all__ = [
    "AlexNet",
    "ExplainableAlexNet",
    "Mclr_CrossEntropy",
    "MLP_MNIST",
    "MLP_CIFAR10",
    "MLP_CIFAR100",
    "Resnet18",
    "Resnet34",
    "Resnet50",
    "Resnet101",
    "Resnet152",
    "SwinTransformer",
    "VisionTransformer",
    "VGG11","VGG13","VGG16","VGG19",
    "MobileNet",
]