from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization
def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        transforms.RandomHorizontalFlip(p=0.5),
                        HistogramEqualization(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],[0.5])
                    ])