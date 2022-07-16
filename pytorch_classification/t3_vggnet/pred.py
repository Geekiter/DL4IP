import torch
from PIL import Image
from torchvision.transforms import transforms

from pytorch_classification.t3_vggnet.model import vgg


def predict(model_path, img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])
    classes = ("plane", 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'horse', 'ship', 'truck')
    model = vgg(num_classes=len(classes))
    model.load_from_checkpoint(model_path)
    im = Image.open(img_path)
    im = transform(im)
    im = torch.unsqueeze(im, dim=0)
    with torch.no_grad():
        outputs = model(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])
