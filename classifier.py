import numpy as np
from collections import Counter
from torch import nn

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

import torchvision.transforms.functional as FF
import torchvision.transforms as transforms


class SquarePad:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])

        if max_wh < self.size:
            max_wh = self.size

        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return FF.pad(image, padding, 0, 'constant')


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Embedder(nn.Module):
    def __init__(self, backbone=torchvision.models.mobilenet_v2,
                 num_frozen_blocks=17
                ):
        super().__init__()
        self.backbone = backbone(pretrained=True)
        self.backbone.classifier = Identity()
        for child in self.backbone.features[:num_frozen_blocks]:
            for params in child.parameters():
                params.requires_grad = False

    def embed(self, x):
        with torch.no_grad():
            return self.backbone(x)

    def forward(self, x):
        return self.backbone(x)


val_transform = transforms.Compose([
    SquarePad(1),
    transforms.Resize((248, 248)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class DishClassifier(nn.Module):
    def __init__(self, embedder, emb_size, n_classes):
        super(DishClassifier, self).__init__()
        self.embedder = embedder

        for param in self.embedder.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(emb_size, n_classes)

    def eval(self):
        super().eval()
        self.embedder.eval()

    def get_params(self):
        return self.linear.parameters()

    def forward(self, x):
        y = self.linear(self.embedder.embed(x))
        return y
