import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models
# from resnest.torch import resnest50

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# svhn --> handcraft
# usps --> resnest

class DANNCNN(nn.Module):

    def __init__(self):
        super(DANNCNN, self).__init__()
        # self.feature_extractor = models.resnet50(weights=None)
        # self.num_features = self.feature_extractor.fc.in_features
        # self.feature_extractor.fc = nn.Identity()

        self.num_features = 2*2*128

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5), # 24
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2), # 12
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=5), # 8
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.MaxPool2d(2), # 4
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3) # 2
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(self.num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature_extractor(input_data)
        # num_features=2048, feature = (128, 2048)
        feature = feature.view(-1, self.num_features)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def get_features(self, input_data):
        features = self.feature_extractor(input_data)
        features = features.view(-1, self.num_features)
        return features



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.num_features = 2*2*128

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5), # 24
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2), # 12
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=5), # 8
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.MaxPool2d(2), # 4
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3) # 2
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(self.num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data):
        feature = self.feature_extractor(input_data)
        feature = feature.view(-1, self.num_features)
        
        class_output = self.class_classifier(feature)

        return class_output