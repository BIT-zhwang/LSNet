import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch

class vgg16(torch.nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.features = models.vgg16_bn(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        out=self.classifier(x)
        return out

class lenet5(torch.nn.Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.conv1=torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2=torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=torch.nn.Linear(in_features=16 * 5 * 5,out_features=120)
        self.fc2=torch.nn.Linear(in_features=120, out_features=84)
        self.fc3=torch.nn.Linear(in_features=84, out_features=10)
        self.fc4 = torch.nn.Linear(in_features=10, out_features=2)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # reshape tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x