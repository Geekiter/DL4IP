import platform

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import transforms

from pytorch_classification.t2_alexnet.model import AlexNet


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    model = model.to(device)
    summary(model, (3, 224, 224))


def train():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    train_loader = DataLoader(train_set, batch_size=36, shuffle=True)
    val_set = datasets.CIFAR10(root="./data", train=False, download=False, transform=val_tf)
    val_loader = DataLoader(val_set, batch_size=5000, shuffle=False)

    max_epoch = 1
    accelerator = "cpu"

    if platform.system().lower() == "linux":
        max_epoch = 50
        accelerator = "gpu"
    trainer = Trainer(max_epochs=max_epoch, accelerator=accelerator, log_every_n_steps=40)
    model = AlexNet()
    trainer.fit(model, train_loader)
    trainer.validate(model, val_loader)
