
import cv2
from utils.face_utils import extract_landmarks
import numpy as np
from torchvision.transforms import functional as F
class ExtractLandmarks(object):
    def __call__(self, image):
        c,w,h = image.shape
        faces=cv2.UMat(np.array([[0,0,w,h]], dtype=np.uint8))
        landmarks = extract_landmarks(image,faces)
        landmarks = landmarks.squeeze(0)
        landmarks = F.normalize(landmarks,[0.5],[0.5])
        return image,landmarks
    def __repr__(self):
        return self.__class__.__name__ + '()'