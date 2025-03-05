import argparse
import cv2
import sys
import numpy as np
import intangles_insightface
from intangles_insightface.app import FaceAnalysis
from intangles_insightface.data import get_image as ins_get_image


parser = argparse.ArgumentParser(description='intangles_insightface gender-age test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
args = parser.parse_args()

app = FaceAnalysis(allowed_modules=['detection', 'genderage'])
app.prepare(ctx_id=args.ctx, det_size=(640,640))

img = ins_get_image('t1')
faces = app.get(img)
assert len(faces)==6
for face in faces:
    print(face.bbox)
    print(face.sex, face.age)

