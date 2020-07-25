
import torch
from utils.image_utils import extract_hog


class HOGFeatureExtraction(object):
    def __init__(self,multichannel=True):
      self.multichannel = multichannel
    def __call__(self, image):
        return extract_hog(image,self.multichannel)
    def __repr__(self):
        return self.__class__.__name__ + '()'