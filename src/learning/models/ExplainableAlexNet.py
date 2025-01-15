import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class ExplainableAlexNet(nn.Module):
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
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=96, kernel_size=11, stride=4) #  (b x 96 x 55 x 55)
        self.lr_norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2) # section 3.3
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=channels, stride=1) # (b x 96 x 27 x 27)
        self.conv2 = nn.Conv2d(96, 256, 1, padding=2) # (b x 256 x 27 x 27)
        self.lr_norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=channels, stride=1) # (b x 256 x 13 x 13)
        self.conv3 = nn.Conv2d(256, 384, channels, padding=1) # (b x 384 x 13 x 13)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, channels, padding=1) # (b x 384 x 13 x 13)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, channels, padding=1) # (b x 256 x 13 x 13)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=channels, stride=1) # (b x 256 x 6 x 6)
        
        # classifier is just a name for linear layers
        self.fc_dense_1 = nn.Linear(in_features=(256*15*15), out_features=4096)
        self.fc_bn_1  = nn.BatchNorm1d(4096, track_running_stats=False)
        self.fc_relu_1 = nn.ReLU()
        self.fc_drop_1 = nn.Dropout(p=0.2, inplace=False)
        
        self.fc_dense_2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc_bn_2 = nn.BatchNorm1d(4096, track_running_stats=False)
        self.fc_relu_2 = nn.ReLU()
        self.fc_drop_2 = nn.Dropout(p=0.2, inplace=False)
        
        self.fc_dense_3 = nn.Linear(in_features=4096, out_features=num_classes)
        self.fc_bn_3 = nn.BatchNorm1d(num_classes, track_running_stats=False)
        self.fc_drop_3 = nn.Dropout(p=0.2, inplace=False)

        self.init_bias()  # initialize bias

    def init_bias(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv3.bias, 0)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv4.bias, 0)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv5.bias, 0)
        
        nn.init.constant_(self.conv2.bias, 1)
        nn.init.constant_(self.conv4.bias, 1)
        nn.init.constant_(self.conv5.bias, 1)

    def forward(self, x, explain = False):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        cv_layer_1 = self.pool1(self.relu1(self.lr_norm1(self.conv1(x))))
        cv_layer_2 = self.pool2(self.relu2(self.lr_norm2(self.conv2(cv_layer_1))))
        cv_layer_3 = self.relu3(self.conv3(cv_layer_2))
        cv_layer_4 = self.relu4(self.conv4(cv_layer_3))
        cv_layer_5 = self.pool5(self.relu5(self.conv5(cv_layer_4)))
        
        x = cv_layer_5.view(-1, 256 * 15 * 15)  # reduce the dimensions for linear layer input
        
        fc_layer_1 = self.fc_drop_1(self.fc_relu_1(self.fc_bn_1(self.fc_dense_1(x))))
        fc_layer_2 = self.fc_drop_2(self.fc_relu_2(self.fc_bn_2(self.fc_dense_2(fc_layer_1))))
        fc_layer_3 = self.fc_drop_3(self.fc_bn_3(self.fc_dense_3(fc_layer_2)))
        
        if explain == True:
            return {
                'cv_layer_1' :cv_layer_1, 
                'cv_layer_2' :cv_layer_2, 
                'cv_layer_3' :cv_layer_3, 
                'cv_layer_4' :cv_layer_4, 
                'cv_layer_5' :cv_layer_5,
                'fc_layer_1' :fc_layer_1, 
                'fc_layer_2' :fc_layer_2, 
                'fc_layer_3' :fc_layer_3
                }
        else:
            return fc_layer_3
    
    def load_weights(self, file):
        self.conv1.weight.data = file["net.conv1.weight"]
        self.conv2.weight.data = file["net.conv2.weight"]
        self.conv3.weight.data = file["net.conv3.weight"]
        self.conv4.weight.data = file["net.conv4.weight"]
        self.conv5.weight.data = file["net.conv5.weight"]
        self.conv1.bias.data = file["net.conv1.bias"]
        self.conv2.bias.data = file["net.conv2.bias"]
        self.conv3.bias.data = file["net.conv3.bias"]
        self.conv4.bias.data = file["net.conv4.bias"]
        self.conv5.bias.data = file["net.conv5.bias"]
        
        self.fc_dense_1.weight.data = file["classifier.fc_layer1.weight"]
        self.fc_dense_2.weight.data = file["classifier.fc_layer_2.weight"]
        self.fc_dense_3.weight.data = file["classifier.fc_layer_3.weight"]
        self.fc_bn_1.weight.data = file["classifier.fc_bn_1.weight"]
        self.fc_bn_2.weight.data = file["classifier.fc_bn_2.weight"]
        self.fc_bn_3.weight.data = file["classifier.fc_bn_3.weight"]
        self.fc_dense_1.bias.data = file["classifier.fc_layer1.bias"]
        self.fc_dense_2.bias.data = file["classifier.fc_layer_2.bias"]
        self.fc_dense_3.bias.data = file["classifier.fc_layer_3.bias"]
        self.fc_bn_1.bias.data = file["classifier.fc_bn_1.bias"]
        self.fc_bn_2.bias.data = file["classifier.fc_bn_2.bias"]
        self.fc_bn_3.bias.data = file["classifier.fc_bn_3.bias"]
            