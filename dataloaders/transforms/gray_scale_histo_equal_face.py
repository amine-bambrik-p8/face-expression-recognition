from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization
from dataloaders.transforms.detect_faces import DetectFaces
import imgaug.augmenters as iaa
import numpy as np
import torchvision.transforms.functional as F

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.HistogramEqualization(),
        ])
    def __call__(self, img):
        img = np.array(img)
        return F.to_pil_image(F.to_tensor(self.aug.augment_image(img)))
def transform():
    return transforms.Compose([
                        transforms.Grayscale(1),
                        DetectFaces(),
                        ImgAugTransform(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],[0.5]),
                    ])