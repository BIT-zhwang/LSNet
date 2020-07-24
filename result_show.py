from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os 
from tqdm import tqdm

files=os.listdir("./logs/stage2")
for file in tqdm(files):
    regions=[]
    with open(os.path.join("./logs/stage2",file),"r") as f:
        lines=f.readlines()
        for line in lines:
            regions.append([int(loc) for loc in line[1:-2].split(", ")])
    #print(regions)
    rects=np.array(regions)
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)
    #print(pick)

    img=cv2.imread(os.path.join("./data/JPEGImages/",file[:-10]+".JPG"))
    for region in pick:
        cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (255, 0, 0), thickness=2)
    cv2.imwrite(os.path.join("./results/stage2/",file[:-10]+".JPG"),img)
