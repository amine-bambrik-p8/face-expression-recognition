from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization

def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        HistogramEqualization(),
                        transforms.RandomRotation(30),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.CenterCrop(40),
                        transforms.Resize(48),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],[0.5]),
                    ])