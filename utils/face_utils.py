import torch
import torchvision
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2
import os
import matplotlib.pyplot as plt

def detect_faces(image_tensor):
    haarcascade = "haarcascade_frontalface_alt2.xml"
    detector = cv2.CascadeClassifier(haarcascade)
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = cv2.imread(pic)
    faces = detector.detectMultiScale(image)
    return faces

def extract_landmarks(image_tensor,faces):
    # Detect landmarks on "image_gray"
    LBFmodel = "lbfmodel.yaml"
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    _, landmarks = landmark_detector.fit(image, faces)
    return torch.tensor(landmarks)