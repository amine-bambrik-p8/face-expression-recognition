import cv2
from utils.face_utils import detect_faces
import torchvision.transforms.functional as F
class DetectFaces(object):
    def __call__(self, pil_image):
        pil_image = F.to_grayscale(pil_image,3) 
        image = F.to_tensor(pil_image)
        faces = detect_faces(image)
        if len(faces) == 0:
            return F.to_grayscale(pil_image,1)
        face = faces[0]
        cropped_image = image[:,face[0]:face[2], face[1]:face[3]]
        cropped_pil_image = F.to_pil_image(cropped_image, mode=None)
        c,w,h = image.shape
        pil_image=F.resize(cropped_pil_image,(w,h))
        pil_image = F.to_grayscale(pil_image,1) 
        return pil_image
    def __repr__(self):
        return self.__class__.__name__ + '()'