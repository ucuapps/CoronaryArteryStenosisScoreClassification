from torch import nn
from torchvision import models
import torch

class ShuffleNetv2(nn.Module):

    def __init__(self, n_classes=2):
        super(ShuffleNetv2, self).__init__()
        self.model =  models.shufflenet_v2_x1_0(pretrained=True)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)
        self.model.eval()


    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

if __name__ == '__main__':
    PATH = '../model_model_34_val_f1=0.9360136.pth'
