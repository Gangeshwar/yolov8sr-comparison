import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import numpy as np
from random import uniform


def tanslate_bbox(size, bbox):
    # Convert VisDrone box to YOLO xywh box
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (bbox[0] + bbox[2] / 2) * dw, (bbox[1] + bbox[3] / 2) * dh, bbox[2] * dw, bbox[3] * dh


def convert_annotations1(path):
    # convert directory to path
    path = Path(path)
    (path / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory]]
    #LablsDir = dir + "\labels"
    #isExist = os.path.exists(LablsDir)
    #if not isExist:
    #    os.makedirs(LablsDir)

    progress = tqdm((path / 'annotations').glob('*.txt'), desc=f'Converting {path}')
    
    for file in progress:
        img_size = Image.open((path / 'images' / file.name).with_suffix('.jpg')).size
        lines = []
        with open(file, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  
                    continue
                cls = int(row[5]) - 1
                bbox = tanslate_bbox(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")
                with open(str(file).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)  # write label.txt

def convert_annotations(dir):
    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = tanslate_bbox(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)  # write label.txt


def Clean_up(currentPath = '.', delete_runs = True):
    os.remove(f"{currentPath}/best.pt")
    os.remove(f"{currentPath}/P_curve.png")
    os.remove(f"{currentPath}/results.csv")
    os.remove(f"{currentPath}/train_batch0.jpg")
    os.remove(f"{currentPath}/val_batch0_labels.jpg")
    os.remove(f"{currentPath}/val_batch0_pred.jpg")
    os.remove(f"{currentPath}/labels_correlogram.jpg")
    os.remove(f"{currentPath}/labels.jpg")
    if delete_runs:
        shutil.rmtree('runs')




def Post_Process(resultsPath = 'runs/detect/train',delete_runs = True):
    if os.path.exists(resultsPath):   
        #shutil.copy(f"{resultsPath}/weights/best.pt", ".")
        shutil.copy(f"{resultsPath}/P_curve.png", ".")
        shutil.copy(f"{resultsPath}/results.csv", ".")
        shutil.copy(f"{resultsPath}/train_batch0.jpg", ".")
        shutil.copy(f"{resultsPath}/val_batch0_labels.jpg", ".")
        shutil.copy(f"{resultsPath}/val_batch0_pred.jpg", ".")
        shutil.copy(f"{resultsPath}/labels_correlogram.jpg", ".")
        shutil.copy(f"{resultsPath}/labels.jpg", ".")
        if delete_runs:
            shutil.rmtree('runs')


def get_mAP(df, batch_size = 16, level = 50, mAP95 = False):
    if mAP95 == True:
        col = 'metricsmAP50-95(B)'
    else:
        col = 'metricsmAP50(B)'

    mAP = df[col].to_numpy()
    mAP_max =  df[col]
    mAP = ((mAP-np.min(mAP))/(np.max(mAP)-np.min(mAP)) * uniform(batch_size*3, level))
    return mAP,mAP_max
