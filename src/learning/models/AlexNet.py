import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """    
    def __init__(self, channels=3, num_classes=10, **kwargs):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        if channels == 3:
            conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=11, stride=4, padding=2) #  (b x 96 x 55 x 55)
            lr_norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2) # section 3.3
            relu1 = nn.ReLU()
            pool1 = nn.MaxPool2d(kernel_size=channels, stride=2) # (b x 96 x 27 x 27)
            conv2 = nn.Conv2d(64, 192, 5, padding=2) # (b x 256 x 27 x 27)
            lr_norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
            relu2 = nn.ReLU()
            pool2 = nn.MaxPool2d(kernel_size=channels, stride=2) # (b x 256 x 13 x 13)
            conv3 = nn.Conv2d(192, 256, kernel_size=channels, padding=1) # (b x 384 x 13 x 13)
            relu3 = nn.ReLU()
            conv4 = nn.Conv2d(256, 384, kernel_size=channels, padding=1) # (b x 384 x 13 x 13)
            relu4 = nn.ReLU()
            conv5 = nn.Conv2d(384, 256, kernel_size=channels, padding=1) # (b x 256 x 13 x 13)
            relu5 = nn.ReLU()
            pool5 = nn.MaxPool2d(kernel_size=channels, stride=2) # (b x 256 x 6 x 6)
            
            # classifier is just a name for linear layers
            fc_layer_1 = nn.Linear(in_features=(256*6*6), out_features=4096)
            fc_bn_1  = nn.BatchNorm1d(4096, track_running_stats=False)
            fc_relu_1 = nn.ReLU()
            fc_drop_1 = nn.Dropout(p=0.2, inplace=False)
            
            fc_layer_2 = nn.Linear(in_features=4096, out_features=4096)
            fc_bn_2 = nn.BatchNorm1d(4096, track_running_stats=False)
            fc_relu_2 = nn.ReLU()
            fc_drop_2 = nn.Dropout(p=0.2, inplace=False)
            
            fc_layer_3 = nn.Linear(in_features=4096, out_features=num_classes)
            fc_bn_3 = nn.BatchNorm1d(num_classes, track_running_stats=False)
            fc_drop_3 = nn.Dropout(p=0.2, inplace=False)
            
            self.view_point = 256*6*6
        
        elif channels == 1:
            conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=11, stride=4) #  (b x 96 x 55 x 55)
            lr_norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2) # section 3.3
            relu1 = nn.ReLU()
            pool1 = nn.MaxPool2d(kernel_size=channels, stride=1) # (b x 96 x 27 x 27)
            conv2 = nn.Conv2d(64, 192, 1, padding=2) # (b x 256 x 27 x 27)
            lr_norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
            relu2 = nn.ReLU()
            pool2 = nn.MaxPool2d(kernel_size=channels, stride=1) # (b x 256 x 13 x 13)
            conv3 = nn.Conv2d(192, 256, channels, padding=1) # (b x 384 x 13 x 13)
            relu3 = nn.ReLU()
            conv4 = nn.Conv2d(256, 384, channels, padding=1) # (b x 384 x 13 x 13)
            relu4 = nn.ReLU()
            conv5 = nn.Conv2d(384, 256, channels, padding=1) # (b x 256 x 13 x 13)
            relu5 = nn.ReLU()
            pool5 = nn.MaxPool2d(kernel_size=channels, stride=1) # (b x 256 x 6 x 6)
            
            # classifier is just a name for linear layers
            fc_layer_1 = nn.Linear(in_features=(256*6*6), out_features=4096)
            fc_bn_1  = nn.BatchNorm1d(4096, track_running_stats=False)
            fc_relu_1 = nn.ReLU()
            fc_drop_1 = nn.Dropout(p=0.2, inplace=False)
            
            fc_layer_2 = nn.Linear(in_features=4096, out_features=4096)
            fc_bn_2 = nn.BatchNorm1d(4096, track_running_stats=False)
            fc_relu_2 = nn.ReLU()
            fc_drop_2 = nn.Dropout(p=0.2, inplace=False)
            
            fc_layer_3 = nn.Linear(in_features=4096, out_features=num_classes)
            fc_bn_3 = nn.BatchNorm1d(num_classes, track_running_stats=False)
            fc_drop_3 = nn.Dropout(p=0.2, inplace=False)
            
            self.view_point = 256*6*6
            
        
        self.features = nn.Sequential(OrderedDict([
            ('conv1', conv1), # layer 1
            ('lr_norm1', lr_norm1), 
            ('relu1', relu1), 
            ('pool1', pool1), 
            ('conv2', conv2), # layer 2
            ('lr_norm2', lr_norm2), 
            ('relu2', relu2), 
            ('pool2', pool2), 
            ('conv3', conv3), # layer 3
            ('relu3', relu3), 
            ('conv4', conv4), # layer 4
            ('relu4', relu4), 
            ('conv5', conv5), # layer 5
            ('relu5', relu5), 
            ('pool5', pool5)
        ]))
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('fc_layer1', fc_layer_1), 
            ('fc_bn_1', fc_bn_1), 
            ('fc_relu_1', fc_relu_1), 
            ('fc_drop_1', fc_drop_1), # layer 1
            ('fc_layer_2', fc_layer_2), 
            ('fc_bn_2', fc_bn_2), 
            ('fc_relu_2', fc_relu_2), 
            ('fc_drop_2', fc_drop_2), # layer 2
            ('fc_layer_3', fc_layer_3), 
            ('fc_bn_3', fc_bn_3),
            ('fc_drop_3', fc_drop_3) # layer 3
        ]))
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        try:
            nn.init.constant_(self.features[4].bias, 1)
            nn.init.constant_(self.features[10].bias, 1)
            nn.init.constant_(self.features[12].bias, 1)
        except:
            pass

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.view_point)  # reduce the dimensions for linear layer input
        return self.classifier(x)