

import torch
from skimage.feature import hog
from skimage import data, exposure

def extract_hog(image_tensor,multichannel=True):
    image = image_tensor.permute(1, 2, 0).numpy()
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(10,10),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    return torch.tensor(hog_image)
