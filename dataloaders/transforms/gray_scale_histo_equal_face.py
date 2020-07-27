from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization
from dataloaders.transforms.detect_faces import DetectFaces


def transform():
    return transforms.Compose([
                    transforms.Grayscale(1),
                    HistogramEqualization(),
                    transforms.Grayscale(3),
                    DetectFaces(),
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5),(0.5))
                ])