import torch
import torchvision
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2
import matplotlib.pyplot as plt
class FeatureExtraction(object):
    def __call__(self, pic):
        return 
    def __repr__(self):
        return self.__class__.__name__ + '()'
transform = transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                ])
train_dataset = torchvision.datasets.ImageFolder(
            root="data/FER/train",
            transform=transform
        )
dataloader=DataLoader(train_dataset,batch_size=4,shuffle=True)
dataiter = iter(dataloader)
images,targets = next(dataiter)
#pic = train_dataset.imgs[0][0];
image = (images[0].permute(1, 2, 0).numpy() * 255).astype('uint8')
#h,w,c = image.shape
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#image = cv2.imread(pic)
haarcascade= "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(haarcascade)
faces = detector.detectMultiScale(image)
print("WTF")

for face in faces:
    #     save the coordinates in x, y, w, d variables
    (x,y,w,d) = face
    # Draw a white coloured rectangle around each face using the face's coordinates
    # on the "image_template" with the thickness of 2 
    cv2.rectangle(image,(x,y),(x+w, y+d),(255, 255, 255), 2)

LBFmodel = "lbfmodel.yaml"
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# Detect landmarks on "image_gray"
_, landmarks = landmark_detector.fit(image, faces)
print(landmarks)
for landmark in landmarks:
    for x,y in landmark[0]:
		# display landmarks on "image_cropped"
		# with white colour in BGR and thickness 1
        cv2.circle(image, (x, y), 1, (255, 255, 255), 1)
plt.axis("off")
plt.imshow(image)
plt.show()
