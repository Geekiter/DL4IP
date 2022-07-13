import torch
from PIL import Image
from torchvision.transforms import transforms

from pytorch_classification.t1_lenet.model import LeNet


def predict():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    classes = ("plane", 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'horse', 'ship', 'truck')
    model = LeNet()
    model.load_state_dict(torch.load('LeNet.pth'))
    im = Image.open('img.jpg')
    im = transform(im)
    im = torch.unsqueeze(im, dim=0)
    with torch.no_grad():
        outputs = model(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])
