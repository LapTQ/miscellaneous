import cv2
import numpy as np

PREV_NAME = '20221122071809888_623'
NEXT_NAME = '20221122071809888_637'


def save(next_boxes, labels, H, W):

    next_boxes = np.float32(next_boxes)
    next_boxes[:, 2:] = next_boxes[:, 2:] - next_boxes[:, :2]
    next_boxes[:, :2] += next_boxes[:, 2:] / 2.
    next_boxes[:, 0] /= W
    next_boxes[:, 1] /= H
    next_boxes[:, 2] /= W
    next_boxes[:, 3] /= H
    
    buf = []
    for label, box in zip(labels, next_boxes):
        if box[2] >= 15 / W and box[3] >= 15 / H:
            buf.append(f'{label} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}')
    with open(f'label_lap/{NEXT_NAME}.txt', 'w') as f:
        print('\n'.join(buf), file=f)
        print('[INFO] saved', NEXT_NAME)
    


def of():

    prev = cv2.imread(f'images_lap/{PREV_NAME}.jpg')
    next = cv2.imread(f'images_lap/{NEXT_NAME}.jpg')

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    assert prev_gray.shape == next_gray.shape, f'got {prev_gray.shape} and {next_gray.shape}'

    H, W = prev_gray.shape

    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, pyr_scale=0.8, levels=20, winsize=30, iterations=10, poly_n=5, poly_sigma=1.2, flags=0)

    with open(f'label_lap/{PREV_NAME}.txt', 'r') as f:
        prev_boxes = f.readlines()
        prev_boxes = [l[:-1] for l in prev_boxes]
        prev_boxes = [[eval(e) for e in l.split()] for l in prev_boxes]

    prev_boxes = np.array(prev_boxes)
    labels = np.int32(prev_boxes[:, 0])
    prev_boxes = prev_boxes[:, 1:]

    prev_boxes[:, :2] -= prev_boxes[:, 2:4] / 2.
    prev_boxes[:, 2:4] += prev_boxes[:, :2]
    prev_boxes[:, 0] *= W
    prev_boxes[:, 1] *= H
    prev_boxes[:, 2] *= W
    prev_boxes[:, 3] *= H

    next_boxes = np.empty_like(prev_boxes, dtype='int32')
    for i in range(len(prev_boxes)):
        
        box = prev_boxes[i]    
        box = np.int32(box)  
        
        cv2.rectangle(prev, box[:2], box[2:], color=(0, 255, 0), thickness=1)

        x1, y1, x2, y2 = box
        x1 = min(x1, W - 1)
        x2 = min(x2, W - 1)
        y1 = min(y1, H - 1)
        y2 = min(y2, H - 1)
        
        next_boxes[i, 0] = np.clip(x1 + flow[y1, x1, 0], 0, W - 1)
        next_boxes[i, 1] = np.clip(y1 + flow[y1, x1, 1], 0, H - 1)
        next_boxes[i, 2] = np.clip(x2 + flow[y2, x2, 0], 0, W - 1)
        next_boxes[i, 3] = np.clip(y2 + flow[y2, x2, 1], 0, H - 1)
        
        cv2.rectangle(next, next_boxes[i, :2], next_boxes[i, 2:], color=(0, 255, 0), thickness=1)




    # show_img = np.concatenate([prev, next], axis=1)
    show_img = next

    cv2.namedWindow(NEXT_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(NEXT_NAME, show_img)
    key = cv2.waitKey(0)

    if key == ord('y'):
        save(next_boxes, labels, H, W)

    cv2.destroyAllWindows()
    
    return next_boxes, labels
    

def ho():

    prev = cv2.imread(f'images_lap/{PREV_NAME}.jpg')
    next = cv2.imread(f'images_lap/{NEXT_NAME}.jpg')

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    assert prev_gray.shape == next_gray.shape, f'got {prev_gray.shape} and {next_gray.shape}'

    H, W = prev_gray.shape
    
    with open(f'label_lap/{PREV_NAME}.txt', 'r') as f:
        prev_boxes = f.readlines()
        prev_boxes = [l[:-1] for l in prev_boxes]
        prev_boxes = [[eval(e) for e in l.split()] for l in prev_boxes]

    prev_boxes = np.array(prev_boxes)
    labels = np.int32(prev_boxes[:, 0])
    prev_boxes = prev_boxes[:, 1:]

    prev_boxes[:, :2] -= prev_boxes[:, 2:4] / 2.
    prev_boxes[:, 2:4] += prev_boxes[:, :2]
    prev_boxes[:, 0] *= W
    prev_boxes[:, 1] *= H
    prev_boxes[:, 2] *= W
    prev_boxes[:, 3] *= H

    # next_boxes = np.empty_like(prev_boxes, dtype='int32')
    
    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    kp_prev, dc_prev = detector.detectAndCompute(prev_gray, None)
    kp_next, dc_next = detector.detectAndCompute(next_gray, None)

    matches = matcher.knnMatch(dc_prev, dc_next, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    matching = cv2.drawMatches(prev_gray, kp_prev, next_gray, kp_next, good, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    prev_points = np.zeros((len(good), 2), dtype=np.float32)
    next_points = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
            prev_points[i, :] = kp_prev[match.queryIdx].pt
            next_points[i, :] = kp_next[match.trainIdx].pt

    
    src = []
    dst = []
    for i in range(len(prev_points)):
        if prev_points[i, 0] >= 0:  ###########################
            src.append(prev_points[i])
            dst.append(next_points[i])
    src = np.array(src)
    dst = np.array(dst)
    prev_points = src
    next_points = dst
    

    homo, mask = cv2.findHomography(prev_points, next_points, cv2.RANSAC) # 
    
    next_boxes = cv2.perspectiveTransform(prev_boxes.reshape(-1, 1, 2), homo).reshape(-1, 4).astype('int32')
    for i, box in enumerate(next_boxes):
        corner = np.array([[[box[0], box[1]]], [[box[0], box[3]]], [[box[2], box[3]]], [[box[2], box[1]]]])
        x, y, w, h = cv2.boundingRect(corner)
        
        next_boxes[i] = [
            np.clip(x, 0, W - 1), 
            np.clip(y, 0, H - 1), 
            np.clip(x + w, 0, W - 1), 
            np.clip(y + h, 0, H - 1)
        ]
        cv2.rectangle(next, next_boxes[i, :2], next_boxes[i, 2:], color=(0, 255, 0), thickness=1)

    ret = cv2.warpPerspective(prev, homo, (W, H))

    #show_img = ret
    #show_img = np.clip(ret * 0.5 + next * 0.5, 0, 255).astype('uint8')
    #show_img = matching
    show_img = next

    cv2.namedWindow(NEXT_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(NEXT_NAME, show_img)
    key = cv2.waitKey(0)
    
    if key == ord('y'):
        save(next_boxes, labels, H, W)
        

    cv2.destroyAllWindows()
    
    return next_boxes, labels
    

def combine():
    
    prev = cv2.imread(f'images_lap/{PREV_NAME}.jpg')
    next = cv2.imread(f'images_lap/{NEXT_NAME}.jpg')
    
    H, W = next.shape[:2]
    
    boxes1, labels = of()
    boxes2, labels = ho()
    next_boxes = (boxes1 + boxes2) // 2
    
    for box in next_boxes:
        cv2.rectangle(next, box[:2], box[2:], color=(0, 255, 0), thickness=1)
        
    cv2.namedWindow(NEXT_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(NEXT_NAME, next)
    key = cv2.waitKey(0)
    
    if key == ord('y'):
        save(next_boxes, labels, H, W)
        

    cv2.destroyAllWindows()


if __name__ == '__main__':
    ho()
    # combine()
    
    

