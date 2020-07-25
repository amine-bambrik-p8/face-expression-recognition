
import cv2
from utils.face_utils import extract_landmarks
import numpy as np

class ExtractLandmarks(object):
    def __call__(self, image,width,height):
        faces=cv2.UMat(np.array([[0,0,width,height]], dtype=np.uint8))
        landmarks = extract_landmarks(image,faces)
        return image,landmarks
    def __repr__(self):
        return self.__class__.__name__ + '()'