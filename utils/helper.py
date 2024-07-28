from pathlib import Path
import os
import numpy as np
import cv2
import math
from tqdm import tqdm


def run(dir):

    global show_img
    global black_list

    img_list = sorted(os.listdir(dir))
    nx = math.ceil(math.sqrt(len(img_list))) * 2
    ny = math.ceil(len(img_list) / nx)
    pad = nx * ny - len(img_list)
    collage = []
    
    max_H = 0
    max_W = 0
    for i, name in enumerate(img_list):
        if i % nx == 0:
            print('******** LINE ********', i // nx)
            collage.append([])
        print(i % nx, '\t', name)
        img = cv2.imread(dir + '/' + name)
        max_H = max(max_H, img.shape[0])
        max_W = max(max_W, img.shape[1])
        collage[-1].append(img)
    for i in range(len(collage)):
        for j in range(len(collage[i])):
            collage[i][j] = cv2.resize(collage[i][j], (max_W, max_H))
        if i == len(collage) - 1:
            for _ in range(pad):
                collage[i].append(np.full((max_H, max_W, 3), 255, dtype='uint8'))
        collage[i] = np.concatenate(collage[i], axis=1)
    
    show_img = np.concatenate(collage, axis=0)
    black_list = []
    
    
    def on_mouse(event, x, y, flags, param):
        global show_img
        global black_list
        
        if event == cv2.EVENT_LBUTTONDOWN:
            i = x // max_W
            j = y // max_H
            
            x1 = i * max_W
            y1 = j * max_H
            x2 = (i + 1) * max_W
            y2 = (j + 1) * max_H
            cv2.rectangle(show_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=6)
            cv2.imshow(dir, show_img)
            to_remove = img_list[j * nx + i]
            if to_remove not in black_list:
                black_list.append(to_remove)
            print(f'About to remove image number {j * nx + i}: {dir}/{to_remove}')


    cv2.namedWindow(dir, cv2.WINDOW_NORMAL)
    cv2.imshow(dir, show_img)
    cv2.setMouseCallback(dir, on_mouse)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:
        return
    elif key == ord('e'):
        exit(0)
    else:
        for name in black_list:
            os.system(f'rm {dir}/{name}')
        run(dir)
    
        
if __name__ == '__main__':
    
    big_dirs = [str(dirs) for dirs in Path('.').glob('*') if os.path.isdir(str(dirs))]
    
    for big_dir in big_dirs:
        dirs = [str(dir) for dir in Path(big_dir).glob('*') if os.path.isdir(str(dir))]
        for dir in tqdm(dirs):
            print('=====================', dir, '=======================')
            run(dir)
