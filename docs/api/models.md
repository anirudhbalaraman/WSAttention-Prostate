# Models Reference

## MILModel_3D

```python
class MILModel_3D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mil_mode: str = "att",
        pretrained: bool = True,
        backbone: str | nn.Module | None = None,
        backbone_num_features: int | None = None,
        trans_blocks: int = 4,
        trans_dropout: float = 0.0,
    )
```

**Constructor arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `num_classes` | `int` | — | Number of output classes |
| `mil_mode` | `str` | `"att"` | MIL aggregation mode |
| `pretrained` | `bool` | `True` | Use pretrained backbone weights |
| `backbone` | `str \| nn.Module \| None` | `None` | Backbone CNN (None = ResNet18-3D) |
| `backbone_num_features` | `int \| None` | `None` | Output features of custom backbone |
| `trans_blocks` | `int` | `4` | Number of transformer encoder layers |
| `trans_dropout` | `float` | `0.0` | Transformer dropout rate |

**MIL modes:**

| Mode | Description |
|------|-------------|
| `mean` | Average logits across all patches — equivalent to pure CNN |
| `max` | Keep only the max-probability instance for loss |
| `att` | Attention-based MIL ([Ilse et al., 2018](https://arxiv.org/abs/1802.04712)) |
| `att_trans` | Transformer + attention MIL ([Shao et al., 2021](https://arxiv.org/abs/2111.01556)) |
| `att_trans_pyramid` | Pyramid transformer using intermediate ResNet layers |

**Key methods:**

- `forward(x, no_head=False)` — Full forward pass. If `no_head=True`, returns patch-level features `[B, N, 512]` before transformer and attention pooling (used during attention loss computation).
- `calc_head(x)` — Applies the MIL aggregation and classification head to patch features.

**Example:**

```python
import torch
from src.model.MIL import MILModel_3D

model = MILModel_3D(num_classes=4, mil_mode="att_trans")
# Input: [batch, patches, channels, depth, height, width]
x = torch.randn(2, 24, 3, 3, 64, 64)
logits = model(x)  # [2, 4]
```

## csPCa_Model

```python
class csPCa_Model(nn.Module):
    def __init__(self, backbone: nn.Module)
```

Wraps a pre-trained `MILModel_3D` backbone for binary csPCa prediction. The backbone's feature extractor, transformer, and attention mechanism are reused. The original classification head (`myfc`) is replaced by a `SimpleNN`.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `backbone` | `MILModel_3D` | Frozen PI-RADS backbone |
| `fc_cspca` | `SimpleNN` | Binary classification head |
| `fc_dim` | `int` | Feature dimension (512 for ResNet18) |

**Example:**

```python
import torch
from src.model.MIL import MILModel_3D
from src.model.csPCa_model import csPCa_Model

backbone = MILModel_3D(num_classes=4, mil_mode="att_trans")
model = csPCa_Model(backbone=backbone)

x = torch.randn(2, 24, 3, 3, 64, 64)
prob = model(x)  # [2, 1] — sigmoid probabilities
```

## SimpleNN

```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim: int)
```

A lightweight MLP for binary classification:

```
Linear(input_dim, 256) → ReLU
Linear(256, 128) → ReLU → Dropout(0.3)
Linear(128, 1) → Sigmoid
```

Input: `[B, input_dim]` — Output: `[B, 1]` (probability).
