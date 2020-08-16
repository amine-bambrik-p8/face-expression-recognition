import cv2
from utils.face_utils import detect_faces
import torchvision.transforms.functional as F
class DetectFaces(object):
    def __call__(self, pil_image):
        pil_image = F.to_grayscale(pil_image,3) 
        image = F.to_tensor(pil_image)
        x0,y0,x1,y1 = detect_faces(image)
        cropped_image = image[:,x0:x1,y0:y1]
        print("Hello")
        print(x0,y0,x1,y1)
        cropped_pil_image = F.to_pil_image(cropped_image, mode=None)
        c,w,h = image.shape
        pil_image=F.resize(cropped_pil_image,(w,h),interpolation=PIL.Image.BICUBIC)
        pil_image = F.to_grayscale(pil_image,1) 
        return pil_image
    def __repr__(self):
        return self.__class__.__name__ + '()'