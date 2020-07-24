import os
import shutil
import random
from xml.dom import minidom
import cv2
from tqdm import tqdm
from skimage.transform import pyramid_gaussian
from skimage import img_as_ubyte,io

import warnings
warnings.filterwarnings('ignore')

def extract_xml(xml_path):
    DOMTree = minidom.parse(xml_path)
    annotation = DOMTree.documentElement
    filename = annotation.getElementsByTagName("filename")[0].childNodes[0].data#img name
    objects = annotation.getElementsByTagName("object")
    regions = []
    for object in objects:
        object_class = object.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0]
        ymin = bndbox.getElementsByTagName('ymin')[0]
        xmax = bndbox.getElementsByTagName('xmax')[0]
        ymax = bndbox.getElementsByTagName('ymax')[0]
        region = [0, 0, 0, 0]
        region[0] = int(xmin.childNodes[0].data)
        region[1] = int(ymin.childNodes[0].data)
        region[2] = int(xmax.childNodes[0].data)
        region[3] = int(ymax.childNodes[0].data)
        regions.append(region)
    return filename, regions

def train_test_split(src_path='/data/hard_mining/epoch1', dst_path='/data/train_model', possibility=0.8):
    lbl_list=["positive","negative"]
    for lbl in lbl_list:
        lst=os.listdir(os.path.join(src_path,lbl))
        for temp in lst:
            if random.random()<possibility:
                shutil.copy(os.path.join(src_path,lbl,temp), os.path.join(dst_path,"train",lbl,temp))
            else:
                shutil.copy(os.path.join(src_path,lbl,temp), os.path.join(dst_path,"test",lbl,temp))

def test_pn(crop, regions, positive_th=0.7, negative_th=0.1):
    ious=[]
    for region in regions:
        ious.append(iou(crop, region))
    if all([iou<negative_th for iou in ious]):
        #negative sample
        return -1
    if any([iou>positive_th for iou in ious]):
        #positive sample 
        return 1
    return 0


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):   
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def iou(detection_1, detection_2):
        # Calculate the x-y co-ordinates of the rectangles
        x1_tl = detection_1[0]
        x2_tl = detection_2[0]
        #x1_br = detection_1[0] + detection_1[2]
        x1_br = detection_1[2]
        x2_br = detection_2[2]
        y1_tl = detection_1[1]
        y2_tl = detection_2[1]
        y1_br = detection_1[3]
        y2_br = detection_2[3]
        # Calculate the overlapping Area
        x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
        y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
        overlap_area = x_overlap * y_overlap
        area_1 = (detection_1[2]-detection_1[0]) * (detection_1[3]-detection_1[1])
        area_2 = (detection_2[2]-detection_2[0]) * (detection_2[3]-detection_2[1])
        total_area = area_1 + area_2 - overlap_area
        return overlap_area / float(total_area)


if __name__=='__main__':
    print("generate the epoch1 dataset...")
    xmls_path='./data/Annotations'
    xml_lst=os.listdir(xmls_path)
    p_cnt=0
    n_cnt=0
    for xml in tqdm(xml_lst):
        filename,regions=extract_xml(os.path.join(xmls_path,xml))
        img_path=os.path.join("./data/JPEGImages",filename)
        image=io.imread(img_path)
        sim=cv2.imread(img_path)
        step_size=2
        window_size=16
        scale=1.2
        level=6
        #for region in regions:
        for (i, img) in enumerate(pyramid_gaussian(image, downscale=scale)):
            if i>(level-1):
                break
            img=img_as_ubyte(img)
            for (x, y, window) in sliding_window(img, stepSize=step_size, windowSize=(window_size,window_size)):
                crop=[x,y,x+window_size,y+window_size]
                crop=[int(loc*(scale**i)) for loc in crop]
                window=sim[crop[1]:crop[3],crop[0]:crop[2]]
                r=test_pn(crop,regions,0.5,0.5)
                if r==1:
                    p_cnt+=1
                    #cv2.imwrite(os.path.join("./data/hard_mining/epoch1/positive",str(p_cnt)+".jpg"),window)
                    #io.imsave(os.path.join("./data/hard_mining/epoch1/positive",str(p_cnt)+".jpg"),window) 
                if r==-1 and random.random()<0.001:
                    n_cnt+=1
                    #cv2.imwrite(os.path.join("./data/hard_mining/epoch1/negative",str(n_cnt)+".jpg"),window)
                    #io.imsave(os.path.join("./data/hard_mining/epoch1/negative",str(n_cnt)+".jpg"),window) 
        print(p_cnt,n_cnt)
        if random.random()>0.0:
            break
    #print("split the dataset into two parts...")
    #train_test_split(src_path='./data/hard_mining/epoch1', dst_path='./data/train_model', possibility=0.8)
        
        
             
