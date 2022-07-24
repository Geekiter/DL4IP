import torch
from PIL import Image
from torchvision.transforms import transforms

from pytorch_classification.t3_vggnet.model import vgg
from pytorch_classification.t4_googlenet.model import GoogLeNet


def predict(model_path, img_path):
    model = GoogLeNet()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    classes = ("plane", 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'horse', 'ship', 'truck')

    model.load_from_checkpoint(model_path)
    im = Image.open(img_path)
    im = transform(im)
    im = torch.unsqueeze(im, dim=0)
    with torch.no_grad():
        outputs = model(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])
