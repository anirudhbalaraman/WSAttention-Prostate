
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from monai.utils.module import optional_import
models, _ = optional_import("torchvision.models")



class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear( 256,128),
            nn.ReLU(),
            nn.Dropout(p=0.3),    
            nn.Linear(128, 1),
            nn.Sigmoid()   # since binary classification
        )
    def forward(self, x):
        return self.net(x)

class csPCa_Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc_dim = backbone.myfc.in_features
        self.fc_cspca = SimpleNN(input_dim=self.fc_dim)

    def forward(self, x):
        sh = x.shape
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])
        x = self.backbone.net(x)
        x = x.reshape(sh[0], sh[1], -1)
        x = x.permute(1, 0, 2)
        x = self.backbone.transformer(x)
        x = x.permute(1, 0, 2)
        a = self.backbone.attention(x)
        a = torch.softmax(a, dim=1)
        x = torch.sum(x * a, dim=1)

        x = self.fc_cspca(x)
        return x

