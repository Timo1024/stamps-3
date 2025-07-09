import torch.nn as nn
import torchvision.models as models


class StampEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # remove classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.backbone(x)  # (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # flatten
        x = self.embedding(x)
        return x
