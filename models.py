from torch import nn
from torchvision import models
import torch
# from pretrainedmodels import se_resnext50_32x4d
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from activation_functions import CReLU, Swish

class ResNet18(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)


class EfficientB4(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(EfficientB4, self).__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=n_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b4')
        features_num = self.model._fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)


import torch.nn as nn
from collections import OrderedDict

class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self, n_classes=2):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('prelu1', nn.PReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('prelu3', nn.PReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('prelu5', nn.PReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(1756920, 84)),
            ('prelu6', nn.PReLU()),
            ('f7', nn.Linear(84, n_classes)),
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

class Perceptron(nn.Module):
    def __init__(self, n_classes):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(786432,n_classes)
        self.act = nn.PReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.act(self.fc1(x))
        return x

class MultilayerPerceptron(nn.Module):
    def __init__(self, n_classes):
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(786432,100)
        self.act1 = nn.PReLU()
        self.fc2 = nn.Linear(100, n_classes)
        self.act2 = nn.PReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x

class OlesNetwork(nn.Module):
    def __init__(self, n_classes):
        super(OlesNetwork, self).__init__()

        # Block with big cernels
        self.conv1 = nn.Conv2d(3, 32, (7,7))
        self.crelu1 = Swish()
        self.conv2 = nn.Conv2d(32, 64, (5,5))
        self.crelu2 = Swish()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # fully connected
        self.fc1 = nn.Linear(984064, n_classes)
        self.final_act = nn.PReLU()

    def forward(self, x):
        x = self.pool(self.crelu1(self.conv1(x)))

        x = self.pool(self.crelu2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.final_act(self.fc1(x))
        return x
# class SeResNext50(nn.Module):

#     def __init__(self, n_classes=2, pretrained=True):
#         super(SeResNext50, self).__init__()
#         if pretrained:
#             self.model = se_resnext50_32x4d(pretrained='imagenet')
#         else:
#             self.model = se_resnext50_32x4d()
#         features_num = 204800#self.model.last_linear.in_features
#         self.model.last_linear = nn.Linear(features_num, n_classes)

class EfficientB7(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(EfficientB7, self).__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=n_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b7')
        features_num = self.model._fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)


class ShuffleNetv2(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(ShuffleNetv2, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def forward(self, x):
        return self.model(x)

# class SeResNext50(nn.Module):
#
#     def __init__(self, n_classes=2, pretrained=True):
#         super(SeResNext50, self).__init__()
#         if pretrained:
#             self.model = se_resnext50_32x4d(pretrained='imagenet')
#         else:
#             self.model = se_resnext50_32x4d()
#         features_num = 204800#self.model.last_linear.in_features
#         self.model.last_linear = nn.Linear(features_num, n_classes)
#
#     def forward(self, x):
#         return self.model(x)

class LSTMClassification(nn.Module):
    def __init__(self, n_classes=3):
        super(LSTMClassification, self).__init__()
        # backbone = models.resnet34(pretrained=True)
        backbone = models.shufflenet_v2_x1_0(pretrained=True)
        layers = list(backbone.children())
        self.backbone = nn.Sequential(*layers[:-1])
        # self.backbone = backbone.features
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = layers[-1].in_features
        self.lstm = nn.LSTM(self.embedding_dim, 512)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        preds = []

        for i, segment_images in enumerate(x):
            segment_features = self.backbone(segment_images)
            # segment_features = torch.flatten(segment_features, 1)
            # segment_features = segment_features
            # segment_features = self.pool(segment_features)
            segment_features = segment_features.mean([2, 3])
            lstm_features, _ = self.lstm(segment_features.view(len(segment_features), 1, self.embedding_dim))
            res = self.fc(lstm_features[-1])
            preds.append(res)

        return torch.cat(preds)


class LSTMDeepClassification(nn.Module):
    def __init__(self, n_classes=3):
        super(LSTMDeepClassification, self).__init__()
        # backbone = models.resnet34(pretrained=True)
        backbone = models.shufflenet_v2_x1_0(pretrained=True)
        layers = list(backbone.children())
        self.backbone = nn.Sequential(*layers[:-1])
        # self.backbone = backbone.features
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = layers[-1].in_features
        self.lstm = nn.LSTM(self.embedding_dim, 512, num_layers=2)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        preds = []

        for i, segment_images in enumerate(x):
            segment_features = self.backbone(segment_images)
            # segment_features = torch.flatten(segment_features, 1)
            # segment_features = segment_features
            # segment_features = self.pool(segment_features)
            segment_features = segment_features.mean([2, 3])
            lstm_features, _ = self.lstm(segment_features.view(len(segment_features), 1, self.embedding_dim))
            res = self.fc(lstm_features[-1])
            preds.append(res)

        return torch.cat(preds)


class LSTMDeepResNetClassification(nn.Module):
    def __init__(self, n_classes=3):
        super(LSTMDeepResNetClassification, self).__init__()
        backbone = models.resnet34(pretrained=True)
        layers = list(backbone.children())
        self.backbone = nn.Sequential(*layers[:-1])
        self.embedding_dim = layers[-1].in_features
        self.lstm = nn.LSTM(self.embedding_dim, 512, num_layers=2)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        preds = []

        for i, segment_images in enumerate(x):
            segment_features = self.backbone(segment_images)
            segment_features = segment_features.mean([2, 3])
            lstm_features, _ = self.lstm(segment_features.view(len(segment_features), 1, self.embedding_dim))
            res = self.fc(lstm_features[-1])
            preds.append(res)

        return torch.cat(preds)

class AttentionResNet18(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionResNet18, self).__init__()
        self.M = 512
        self.L = 512

        backbone = models.resnet18(pretrained=pretrained)
        layers = list(backbone.children())
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out


class AttentionResNet34(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionResNet34, self).__init__()
        self.L = 512

        backbone = models.resnet34(pretrained=pretrained)
        self.M = backbone.fc.in_features
        layers = list(backbone.children())
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out

class AttentionShuffleNetV2(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionShuffleNetV2, self).__init__()
        self.L = 512

        backbone = models.shufflenet_v2_x1_0(pretrained=pretrained)
        self.M = backbone.fc.in_features * 256
        layers = list(backbone.children())
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)

        features = features.view(features.size(0), features.size(1) * features.size(2) * features.size(3))
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out

class AttentionSqueezeNet(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionSqueezeNet, self).__init__()
        self.L = 512

        backbone = models.squeezenet1_1(pretrained=pretrained)
        layers = list(backbone.children())
        self.M = 492032
        extractor = nn.Sequential(*layers[:-1])

        self.feature_extractor_part1 = extractor

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)

        features = features.view(features.size(0), features.size(1) * features.size(2) * features.size(3))
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out


class AttentionEfficientB4(nn.Module):
    def __init__(self, n_classes=3, pretrained=True):
        super(AttentionEfficientB4, self).__init__()
        self.L = 512

        backbone = EfficientNet.from_pretrained('efficientnet-b4', num_classes=n_classes)
        self.M = 1792

        backbone._fc = nn.Linear(self.M, n_classes)

        self.feature_extractor_part1 = backbone

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False),
            nn.Softmax(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, n_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        features = self.feature_extractor_part1(x)

        features = features.view(features.size(0), features.size(1) * features.size(2) * features.size(3))
        features = features.view(batch_size, features.size(0), -1)

        attention = self.attention(features)
        out = (attention * features).sum(1)
        out = self.classifier(out)

        return out
