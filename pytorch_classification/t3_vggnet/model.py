from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT


class VGG(pl.LightningModule):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.output = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self.train_acc = torchmetrics.Accuracy()

    def forward(self, x, *args, **kwargs) -> Any:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.output(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)

        self.train_acc(outputs, targets)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)

        self.log("training loss", loss)

    def training_step_end(self, batch_parts):
        # predictions from each GPU
        predictions = batch_parts["pred"]
        # losses from each GPU
        losses = batch_parts["loss"]

        gpu_0_prediction = predictions[0]
        gpu_1_prediction = predictions[1]

        # do something with both outputs
        return (losses[0] + losses[1]) / 2

    def test_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        self.log("test loss", loss)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "model name not in cfgs dict"
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    return model
