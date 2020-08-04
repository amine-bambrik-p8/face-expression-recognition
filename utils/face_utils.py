import torch
import torchvision
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2
import os
import matplotlib.pyplot as plt

haarcascade = "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(haarcascade)
def detect_faces(image_tensor):
    global detector
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = cv2.imread(pic)
    faces = detector.detectMultiScale(image)
    if len(faces) != 0:
      top, right, bottom, left = faces[0]
      w, h = right-left, top-bottom
      x0, y0 = left, bottom
      x1, y1 = right, top
      x2, x3 = x1-w,  x0+w
      w=x2-x3
      h=y0-y1
      if h>10 and w>10:
        return x3,y1,x2,y0
    return 0,0,48,48

LBFmodel = "lbfmodel.yaml"
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)
def extract_landmarks(image_tensor,faces):
    global landmark_detector
    # Detect landmarks on "image_gray"
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    _, landmarks = landmark_detector.fit(image, faces)
    return torch.tensor(landmarks)