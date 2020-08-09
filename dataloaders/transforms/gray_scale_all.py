from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization
from dataloaders.transforms.detect_faces import DetectFaces


def transform():
    return transforms.Compose([
                    transforms.Grayscale(1),
                    DetectFaces(),
                    HistogramEqualization(),
                    transforms.RandomRotation(25),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.CenterCrop(44),
                    transforms.Resize(48),
                    transforms.ToTensor(),
                    transforms.RandomErasing(0.3,(0.15,0.15)),
                    transforms.Normalize([0.5],[0.5]),
                ])