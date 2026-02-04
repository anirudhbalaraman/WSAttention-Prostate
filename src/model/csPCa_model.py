from __future__ import annotations
import torch
import torch.nn as nn
from monai.utils.module import optional_import

models, _ = optional_import("torchvision.models")


class SimpleNN(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for binary classification.

    This network consists of two hidden layers with ReLU activation and a dropout layer,
    followed by a final sigmoid activation for probability output.

    Args:
        input_dim (int): The number of input features.
    """

    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # since binary classification
        )

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, input_dim).

        Returns:
            torch.Tensor: Output probabilities of shape (Batch, 1).
        """
        return self.net(x)


class csPCa_Model(nn.Module):
    """
    Clinically Significant Prostate Cancer (csPCa) risk prediction model using a MIL backbone.

    This model repurposes a pre-trained Multiple Instance Learning (MIL) backbone (originally
    designed for PI-RADS prediction) for binary csPCa risk assessment. It utilizes the
    backbone's feature extractor, transformer, and attention mechanism to aggregate instance-level
    features into a bag-level embedding.

    The original fully connected classification head of the backbone is replaced by a
    custom :class:`SimpleNN` head for the new task.

    Args:
        backbone (nn.Module): A pre-trained MIL model. The backbone must possess the
            following attributes/sub-modules:
            - ``net``: The CNN feature extractor.
            - ``transformer``: A sequence modeling module.
            - ``attention``: An attention mechanism for pooling.
            - ``myfc``: The original fully connected layer (used to determine feature dimensions).

    Attributes:
        fc_cspca (SimpleNN): The new classification head for csPCa prediction.
        backbone: The MIL based PI-RADS classifier.
    """

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
