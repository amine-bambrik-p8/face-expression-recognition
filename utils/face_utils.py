import torch
import torchvision
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2
import os
import matplotlib.pyplot as plt
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
detector_2 = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
haarcascade = "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(haarcascade)
def detect_faces(image_tensor):
    global detector_2
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector_2(image, 2)
    for rect in rects:
      (x, y, w, h) = rect_to_bb(rect)
      #faceAligned=fa.align(image,gray,rect)
      y2=y + h
      x2=x + w
      return x,y,x2,y2
    return 0,0,48,48
def face_align(image_tensor):
    global detector_2
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector_2(image, 2)
    for rect in rects:
      (x, y, w, h) = rect_to_bb(rect)
      faceAligned=fa.align(image,gray,rect)
      return faceAligned
      
LBFmodel = "lbfmodel.yaml"
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)
def extract_landmarks(image_tensor,faces):
    global landmark_detector
    # Detect landmarks on "image_gray"
    image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
    _, landmarks = landmark_detector.fit(image, faces)
    return torch.tensor(landmarks)