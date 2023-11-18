import xml.etree.cElementTree as ET
import sys
from pathlib import Path
import numpy as np

def iou_batch(boxes1, boxes2):
    """
    boxes1: [[x1, y1, x2, y2,...], ...]
    boxes2: [[x1, y1, x2, y2,...], ...]
    return matrix dim: len(boxes1) x len(boxes2)
    """
    boxes1 = np.array(boxes1).reshape(-1, 4)
    boxes2 = np.array(boxes2).reshape(-1, 4)
    boxes1 = np.expand_dims(boxes1, 1)
    boxes2 = np.expand_dims(boxes2, 0)

    xx1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    yy1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xx2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    yy2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
              + (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1]) - wh)
    return (o)

def clean(path):
    tr = ET.parse(path)
    root = tr.getroot()
    face_elements = []
    face_boxes = []
    head_elements = []
    head_boxes = []
    def key(e):
        for sub in e:
            if sub.tag == 'name' and sub.text == 'face':
                    return 10
        return 0
    root[:] = sorted(root, key=key)
    
    for e in root:
        is_cellphone = False
        is_face = False
        is_head = False
        for sub in e:
            if sub.tag == 'name' and sub.text == 'cellphone':
                is_cellphone = True
                break
            elif sub.tag == 'name' and sub.text == 'face':
                is_face = True
                break
            elif sub.tag == 'name' and sub.text == 'head':
                is_head = True
                break
        if is_cellphone:
            for sub in e:
                if sub.tag == 'bndbox':
                    arr = []
                    for subsub in sub:
                        arr.append(int(subsub.text))
                    if (arr[2] - arr[0]) * (arr[3] - arr[1]) >= 40000:
                        root.remove(e)
        elif is_face:
            face_elements.append(e)
            for sub in e:
                if sub.tag == 'bndbox':
                    arr = []
                    for subsub in sub:
                        arr.append(int(subsub.text))
                    face_boxes.append(arr)
        elif is_head:
            head_elements.append(e)
            for sub in e:
                if sub.tag == 'bndbox':
                    arr = []
                    for subsub in sub:
                        arr.append(int(subsub.text))
                    head_boxes.append(arr)
    iou_mat = iou_batch(face_boxes, head_boxes)
    face_idxs, head_idxs = np.where(iou_mat > 0.7)
    for face_idx in face_idxs:
        root.remove(face_elements[face_idx.item()])
    return tr

if __name__ == '__main__':
    paths = list(Path('xml').glob('*'))
    # paths = ['xml/20221212_222345_1_scft_005_obuko_narrow_elecom_1670.xml']
    for path in paths:
        tr = clean(path)
        tr.write(path)
    
