import platform

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import transforms

from pytorch_classification.t9_efficientnet.model import efficientnet_b0


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = efficientnet_b0()
    model = model.to(device)
    summary(model, (3, 224, 224))


def train(model_path=None):
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
    num_workers = 0
    max_epoch = 1
    accelerator = "cpu"
    if platform.system().lower() == "linux":
        max_epoch = 20
        accelerator = "gpu"
        num_workers = 8

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    train_loader = DataLoader(train_set, batch_size=36, shuffle=True, num_workers=num_workers)
    val_set = datasets.CIFAR10(root="./data", train=False, download=False, transform=val_tf)
    val_loader = DataLoader(val_set, batch_size=5000, shuffle=False, num_workers=num_workers)

    trainer = Trainer(max_epochs=max_epoch, accelerator=accelerator, log_every_n_steps=40)

    classes = ("plane", 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'horse', 'ship', 'truck')
    model = efficientnet_b0(num_classes=len(classes))
    if model_path is not None:
        model.load_from_checkpoint(model_path)
    trainer.fit(model, train_loader)
    trainer.validate(model, val_loader)
