import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
class HistogramEqualization(object):
    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        """
        img = np.asarray(pic)
        flat = img.flatten()
        
        hist = self.get_histogram(flat, 256)
        cs = self.cumsum(hist)
        nj = (cs - cs.min()) * 255
        N = cs.max() - cs.min()

        # re-normalize the cdf
        cs = nj / N
        cs = cs.astype('uint8')
        img_new = cs[flat]
        img_new = np.reshape(img_new, img.shape)
        return Image.fromarray(img_new)
    def get_histogram(self,image, bins):
            # array with size of bins, set to zeros
        histogram = np.zeros(bins)
        
        # loop through pixels and sum up counts of pixels
        for pixel in image:
            histogram[pixel] += 1
        
        # return our final result
        return histogram
    def cumsum(self,a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def __repr__(self):
        return self.__class__.__name__ + '()'