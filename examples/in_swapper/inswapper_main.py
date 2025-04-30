import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import intangles_insightface
from intangles_insightface.app import FaceAnalysis
from intangles_insightface.data import get_image as ins_get_image


assert intangles_insightface.__version__>='0.1'

if __name__ == '__main__':
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = intangles_insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)


    img = ins_get_image('t1')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    assert len(faces)==6
    source_face = faces[2]
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("./t1_swapped.jpg", res)
    res = []
    for face in faces:
        _img, _ = swapper.get(img, face, source_face, paste_back=False)
        res.append(_img)
    res = np.concatenate(res, axis=1)
    cv2.imwrite("./t1_swapped2.jpg", res)


