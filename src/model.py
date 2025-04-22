import torch
import torch.nn as nn
import torchvision.models as models

class ImageOnlyNutritionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.efficientnet_b0(pretrained=True)
        self.encoder.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # calories, protein, fat, carbs
        )

    def forward(self, images):
        features = self.encoder(images)
        out = self.regressor(features)
        return out
