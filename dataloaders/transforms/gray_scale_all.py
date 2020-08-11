from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization
from dataloaders.transforms.detect_faces import DetectFaces


def transform():
    return transforms.Compose([
                    transforms.Grayscale(1),
                    HistogramEqualization(),
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(p=0.5),
                    DetectFaces(),
                    transforms.ToTensor(),
                    transforms.RandomErasing(0.2),
                    transforms.Normalize([0.5],[0.5]),
                ])