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
        top, right, bottom, left = faces[0]
        w, h = right-left, top-bottom
        x0, y0 = left, bottom
        x1, y1 = right, top
        x2, x3 = x1-w,  x0+w
        cropped_image = image[:,y1:y0, x2:x3]
        cropped_pil_image = F.to_pil_image(cropped_image, mode=None)
        c,w,h = image.shape
        pil_image=F.resize(cropped_pil_image,(w,h))
        pil_image = F.to_grayscale(pil_image,1) 
        return pil_image
    def __repr__(self):
        return self.__class__.__name__ + '()'