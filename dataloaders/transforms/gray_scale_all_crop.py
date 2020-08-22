from torchvision import transforms
from dataloaders.transforms.histo_equal import HistogramEqualization
from dataloaders.transforms.detect_faces import DetectFaces
import imgaug.augmenters as iaa
import numpy as np
import torchvision.transforms.functional as F

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-30, 30), mode='symmetric'),
            iaa.HistogramEqualization(),
            iaa.Crop(px=(5,9)),
            iaa.Sometimes(0.4,iaa.arithmetic.Cutout(size=0.5,fill_mode="gaussian", fill_per_channel=True)),
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