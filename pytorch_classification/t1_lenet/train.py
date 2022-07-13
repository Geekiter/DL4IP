import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import transforms

from pytorch_classification.t1_lenet.model import LeNet


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    model = model.to(device)
    summary(model, (3, 32, 32))


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=36, shuffle=True)
    val_set = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    val_loader = DataLoader(val_set, batch_size=5000, shuffle=False)
    model = LeNet()
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):
        correct_num = 0
        for step, (inputs, targets) in enumerate(train_loader, start=0):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_function(outputs, targets)

            loss.backward()
            optimizer.step()

            predict = torch.max(outputs, dim=1)[1]
            correct_num += torch.eq(predict, targets).sum().item()
        print("%d, acc: %.3f" % (epoch + 1, correct_num / len(train_loader)))

        correct_num = 0
        model.eval()
        with torch.no_grad():
            for index, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                predict = torch.argmax(outputs, 1)
                correct_num += torch.eq(predict, targets).sum().item()
            print("%d, acc: %.3f" % (epoch + 1, correct_num / len(val_loader)))

    save_path = "./LeNet.pth"
    torch.save(model.state_dict(), save_path)
