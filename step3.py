from PIL import Image
from torchvision import datasets,transforms
import argparse
import torch
from cores.net import vgg16,lenet5
import os
from tqdm import tqdm
from xml.dom import minidom
from skimage import img_as_ubyte,io
from skimage.transform import pyramid_gaussian
import cv2

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

def test_pn(crop, regions, positive_th=0.5, negative_th=0.5):
    ious=[]
    for region in regions:
        ious.append(iou(crop, region))
    if all([iou<negative_th for iou in ious]):
        #negative sample
        return 0
    if any([iou>positive_th for iou in ious]):
        #positive sample 
        return 1
    return -1

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Step3 validation!!!')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device=torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform=transforms.Compose([transforms.Resize((32,32), interpolation=2),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
    model = lenet5().to(device)
    print("Network Structure:",model)
    model.load_state_dict(torch.load('./weights/lenet5_best.pth'))
    model.eval()

    print("Step3: check model ...")
    xmls_path='./data/Annotations'
    xml_lst=os.listdir(xmls_path)
    p_cnt=0
    n_cnt=0
    t_cnt=0
    for xml in xml_lst:
        filename,regions=extract_xml(os.path.join(xmls_path,xml))
        img_path=os.path.join("./data/JPEGImages",filename)
        image=io.imread(img_path)
        pil=Image.open(img_path)
        cv=cv2.imread(img_path)
        step_size=4
        window_size=16
        scale=1.2
        level=6
        results=[]
        #for region in regions:
        for (i, img) in enumerate(pyramid_gaussian(image, downscale=scale)):
            if i>(level-1):
                break
            img=img_as_ubyte(img)
            for (x, y, window) in tqdm(sliding_window(img, stepSize=step_size, windowSize=(window_size,window_size))):
                crop=[x,y,x+window_size,y+window_size]
                crop=[int(loc*(scale**i)) for loc in crop]
                window=cv[crop[1]:crop[3],crop[0]:crop[2]]
                data=pil.crop((crop[0], crop[1], crop[2], crop[3]))#left,top,rght,bottom
                data=transform(data)
                data=data.to(device)
                output = model(data.unsqueeze(0))
                pred = output.max(1, keepdim=True)[1][0][0]#0 or 1
                r=test_pn(crop,regions,0.5,0.2)
                t_cnt+=1
                if pred==1:
                    results.append(crop)
                if r==pred:
                    pass
                else:
                    if r==1:
                        p_cnt+=1
                        #window.save(os.path.join("./data/hard_mining/epoch2/positive",str(p_cnt)+"_2.jpg"))#pil
                        #cv2.imwrite(os.path.join("./data/hard_mining/epoch2/positive",str(p_cnt)+"_2.jpg"),window)
                        #io.imsave(os.path.join("./data/hard_mining/epoch2/positive",str(p_cnt)+".jpg"),window) 
                    if r==0:
                        n_cnt+=1
                        #window.save(os.path.join("./data/hard_mining/epoch2/positive",str(p_cnt)+"_2.jpg"))
                        #cv2.imwrite(os.path.join("./data/hard_mining/epoch2/negative",str(n_cnt)+"_2.jpg"),window)
                        #io.imsave(os.path.join("./data/hard_mining/epoch2/negative",str(n_cnt)+".jpg"),window) 
        with open(os.path.join('./logs',filename[:-4]+"epoch1.txt"),"w") as f:
            for result in results:
                f.write(str(result)+"\n")
        print(p_cnt,n_cnt,t_cnt)
        break
        