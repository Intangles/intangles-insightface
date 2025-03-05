import argparse
import cv2
import sys
import numpy as np
import intangles_insightface
from intangles_insightface.app import FaceAnalysis
from intangles_insightface.data import get_image as ins_get_image

assert intangles_insightface.__version__>='0.1'

parser = argparse.ArgumentParser(description='intangles_insightface app test')
# general
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
args = parser.parse_args()

app = FaceAnalysis()
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))

img = ins_get_image('t1')
faces = app.get(img)
assert len(faces)==6
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
feats = []
for face in faces:
    feats.append(face.normed_embedding)
feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats, feats.T)
print(sims)


