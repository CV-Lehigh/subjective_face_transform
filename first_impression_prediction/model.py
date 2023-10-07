import torch
import torch.nn as nn
import torchvision
import torchvision.models as pretrained_models

class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel, self).__init__()
         
        pretrained_mod = pretrained_models.resnet18(weights=pretrained_models.ResNet18_Weights.DEFAULT)
        in_feature_linear = list(pretrained_mod.children())[-1].in_features
        modules = list(pretrained_mod.children())[:-1]  # delete the classifier.
        self.model = nn.Sequential(*modules)
        
        self.linear_layers = nn.Sequential(nn.Linear(in_feature_linear, 1))
        
    def forward(self, x_cnn, forward_type='train'):
        if x_cnn.size(1) > 3:
            x_cnn = x_cnn.permute(0, 2, 1, 3) # from NHWC to NCHW (3 channels)
        x = self.model(x_cnn)
        x = x.reshape(x_cnn.size(0), -1)        
        return self.linear_layers(x)