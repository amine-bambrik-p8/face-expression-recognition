import cv2
from utils.face_utils import extract_landmarks
import torchvision.transforms.functional as F
class DetectFaces(object):
    def __call__(self, image):
        faces = detect_faces(image)
        c,w,h image.shape
        if len(faces) == 0:
            return image
        face = faces[0]
        cropped_image = image[:,face[0]:face[2], face[1]:face[3]]
        cropped_pil_image F.to_pil_image(cropped_image, mode=None)
        pil_image=F.resize(cropped_pil_image,(w,h))
        return F.to_tensor(pil_image)
    def __repr__(self):
        return self.__class__.__name__ + '()'