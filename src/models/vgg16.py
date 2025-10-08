# models/vgg16.py
import torch.nn as nn
from torchvision import models

def build_vgg16(num_classes: int, img_size: int = 224, freeze_backbone: bool = False):
    """
    VGG16-BN pretrained on ImageNet with a classifier head for `num_classes`.
    Respects `freeze_backbone`: if True, only the classifier head is trainable.
    No other side effects; safe for your current train.py.
    """
    # Load pretrained VGG16 with BatchNorm weights
    vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)

    # Replace the last classifier layer to match your dataset
    in_feats = vgg.classifier[-1].in_features  # 4096
    vgg.classifier[-1] = nn.Linear(in_feats, num_classes)

    # Optional: freeze convolutional 'features' backbone
    if freeze_backbone:
        for p in vgg.features.parameters():
            p.requires_grad = False
        # keep classifier trainable
        for p in vgg.classifier.parameters():
            p.requires_grad = True

    return vgg


# (Optional) helpers if you later want to unfreeze *only* part of the backbone.
# Not used by default (your train.py uses a simple "unfreeze all" toggle),
# but available for future experiments without touching train.py.
def unfreeze_last_conv_blocks(vgg_model: nn.Module, blocks: int = 1):
    """
    Unfreezes the last `blocks` conv blocks of VGG16-BN.
    Use only if you later modify train.py to call this.
    """
    from torch import nn as _nn
    assert 1 <= blocks <= 5
    # Freeze all features first
    for p in vgg_model.features.parameters():
        p.requires_grad = False

    # Find MaxPool indices to split into 5 blocks
    pool_idx = [i for i, m in enumerate(vgg_model.features) if isinstance(m, _nn.MaxPool2d)]
    block_ranges = []
    prev = 0
    for idx in pool_idx:
        block_ranges.append((prev, idx))  # inclusive range
        prev = idx + 1

    # Unfreeze last N blocks
    for (start, end) in block_ranges[-blocks:]:
        for i in range(start, end + 1):
            layer = vgg_model.features[i]
            if hasattr(layer, "parameters"):
                for p in layer.parameters():
                    p.requires_grad = True
    return vgg_model
