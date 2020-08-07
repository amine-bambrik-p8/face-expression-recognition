from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization

def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        HistogramEqualization(),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.RandomErasing(0.5,(0.2,0.2)),
                        transforms.Normalize([0.5],[0.5]),
                    ])