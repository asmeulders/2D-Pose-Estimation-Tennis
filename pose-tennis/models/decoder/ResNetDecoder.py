import torch.nn as nn
import torch.nn.functional as F


class ResNetDecoder(nn.Module):
    def __init__(self, encoder_name="resnet50") -> None:
        super().__init__()

        feature_mapping = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

        self.in_features = feature_mapping[encoder_name]
        self.hidden1 = nn.Linear(self.in_features, 512)
        self.ReLU1 = nn.ReLU()
        self.output = nn.Linear(512, 28) # 14 joints * 2 coordinates

    def forward(self, x):
        x = self.hidden1(x)
        x = self.ReLU1(x)
        x = self.output(x)
        return x