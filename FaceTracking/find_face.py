import dlib
from imutils import face_utils
import os

class FindFace:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        path = "dlib_pretrained_model.dat"
        self.predictor = dlib.shape_predictor(path)

    def getfaces(self, image):
        rects = self.detector(image, 1)
        boxes = []
        try:
            for rect in rects:
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                box = [x, y, x + w, y + h]
                boxes.append(box)
        except:
            ''
        return boxes