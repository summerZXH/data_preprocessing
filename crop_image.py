 #!/usr/bin/python
 # -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random

annopath="/home/zhuxiuhong/src/project/object_detection/faster_rcnn_pytorch/data/INSULATOR2017/Annotations"
imgpath="/home/zhuxiuhong/src/project/object_detection/faster_rcnn_pytorch/data/INSULATOR2017/JPEGImages"
cachedir="/home/zhuxiuhong/src/project/object_detection/faster_rcnn_pytorch/data/INSULATOR2017/cache"
scale = 3


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    # import ipdb 
    # ipdb.set_trace()
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        print obj_struct['name']
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        size = obj.find('size')
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

# rotate
def make_rotate(img,boxes):
    """
    img type numpy size rows,cols,channel
    boxes type list  like [[xmin,ymin,xmax,ymax,label][...][....]]
    the boxes's cordinate is PercentCoords 
    """
    #cv2.imshow('pre_rotate', img)
    #height width
    rows,cols,_ = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    rows_new = cols
    cols_new = rows
    M[0,2] += (cols_new-cols)/2
    M[1,2] += (rows_new-rows)/2
    img = cv2.warpAffine(img,M,(cols_new,rows_new))
    boxes_total = []
    rows_1,cols_1,_ = img.shape
    for ii in boxes:
        box = []
        width = ii[2]-ii[0]
        height = ii[3]-ii[1]
        new_x1 = ii[1]
        new_y1 = cols-ii[2]
        new_x2 = ii[3]
        new_y2 = cols-ii[0]
        box.append(new_x1)
        box.append(new_y1)
        box.append(new_x2)
        box.append(new_y2)
        boxes_total.append(box)
        #cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), (255, 0, 255), 2)  
    return img,boxes_total

def updatexml(ann_path,new):
    updatetree=ET.parse(ann_path)
    print ann_path
    root=updatetree.getroot()
    filename=root.find('filename')
    filename.text=str(new[0])
    #print "====="
    #print new[0]
    #print filename.text
    size=root.find('size')
    width=size.find('width')
    width.text=str(new[1])
    height=size.find('height')
    height.text=str(new[2])
    #import ipdb
    #ipdb.set_trace()
    print('*********************************')
    print new[3]
    num=0
    for obj in updatetree.findall('object'):
        print(int(new[4]))
        if(num<int(new[4])):
            bbox=obj.find('bndbox')
            xmin=bbox.find('xmin')
            xmin.text=str(new[3][num][0])
            ymin=bbox.find('ymin')
            ymin.text=str(new[3][num][1])
            xmax=bbox.find('xmax')
            xmax.text=str(new[3][num][2])
            ymax=bbox.find('ymax')
            ymax.text1=str(new[3][num][3])   
        else:
            root.remove(obj)
        num+=1
    updatetree.write(os.path.join(cachedir,new[0]+'.xml'))

def preprocess():
    filelist=os.listdir(annopath)
    num = 0
    number = 0
    for file in filelist:
        print('number =  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',number)
        number += 1
        flag = False
        ann_path = os.path.join(annopath,file)
        ann = parse_rec(ann_path)
        
        name = os.path.splitext(file)[0]
        jpg = name + '.jpg'
        print jpg
        jpg_path = os.path.join(imgpath,jpg)
        img=cv2.imread(jpg_path)
   
        width = img.shape[1]
        height = img.shape[0]
        # print('origin image width = %d'%width)
        # print('origin image height = %d'%height)
        
        boxes = []
        for i in range(len(ann)):
            x1 = ann[i]['bbox'][0]
            y1 = ann[i]['bbox'][1]
            x2 = ann[i]['bbox'][2]
            y2 = ann[i]['bbox'][3]
            # print(x1,y1,x2,y2)
            boxes.append([x1,y1,x2,y2])
            
        if(width < height):
            img, boxes = make_rotate(img,boxes)  
            width = img.shape[1]
            height = img.shape[0]
            # print('rotate filename---------',jpg)
            # print('After rotate image width = %d'%width)
            # print('After rotate image height = %d'%height)
        resize = cv2.resize(img, (int(width / scale), int(height / scale)))
       
        #import ipdb
        #ipdb.set_trace()
        for index in range(len(boxes)):
            x1 = boxes[index][0]
            y1 = boxes[index][1] 
            x2 = boxes[index][2]
            y2 = boxes[index][3]
            
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            
            for j in range(1):
                ratio = random.uniform(0,1)
                if(center_x < width*1.0/2):
                    if(width*1.0/4 < center_x):
                        #max = (x1-(x2-2*height))*1.0/x1
                        ratio = random.uniform(0,0.5)
                    crop_x1 = int(x1 * (1- ratio))
                    crop_y1 = 0
                    tmp = 2*height + crop_x1
                    if (tmp < width):
                        if(tmp < x2):
                            crop_x2 = x2
                        else:
                            crop_x2 = tmp
                    else:
                        crop_x2 = width
                    crop_y2 = height
                else:
                    if(center_x < width*3.0/4):
                        #max = (x1+2*height-x2)*1.0/(width-x2)
                        ratio = random.uniform(0.5,1)
                    crop_x2 = int(x2 + (width - x2)*ratio)
                    crop_y2 = height
                    tmp = crop_x2 - 2*height
                    if(tmp > 0):
                        if(tmp > x1):
                            crop_x1 = x1
                        else:
                            crop_x1 = tmp
                    else:
                        crop_x1 = 0
                    crop_y1 = 0
                
                # crop  
                crop = img[crop_y1:crop_y2,crop_x1:crop_x2]
                # save crop img and xml
                new_name = (6 - len(str(num)))*'0' + str(num)
                cv2.imwrite(os.path.join(cachedir,new_name+'.jpg'), crop)
                num += 1
                    
                new_boxes = []
                crop_width = crop_x2 - crop_x1
                crop_height = crop_y2 - crop_y1
                
                crop_resize = cv2.resize(crop, (int(crop.shape[1] / scale), int(crop.shape[0] / scale)))
                for item in range(len(boxes)):
                    x1 = boxes[item][0]
                    y1 = boxes[item][1] 
                    x2 = boxes[item][2]
                    y2 = boxes[item][3]
                    rec_x1 = x1 - crop_x1
                    rec_y1 = y1 - crop_y1
                    rec_x2 = x2 - crop_x1
                    rec_y2 = y2 - crop_y1
                    if((crop_x1<x1)and(x1<crop_x2)and(crop_x1<x2)and(x2<crop_x2)):
                        new_boxes.append([rec_x1,rec_y1,rec_x2,rec_y2])
                
                        
                    cv2.rectangle(crop_resize, (int(rec_x1/scale), int(rec_y1/scale)), (int(rec_x2/scale), int(rec_y2/scale)), (0, 255, 0), 2)
                    cv2.rectangle(resize, (int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale)), (255, 255, 0), 2)  
                    cv2.rectangle(resize, (int(crop_x1/scale), int(crop_y1/scale)), (int(crop_x2/scale), int(crop_y2/scale)), (255, 0, 0), 2)  
                   
                # print('rec ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                # print(x1,y1,x2,y2)
                # print(crop_x1,crop_y1,crop_x2,crop_x2)    
                # print(rec_x1,rec_y1,rec_x2,rec_x2)   
                print new_boxes
                count=len(new_boxes)
                #new xml information
                new=[new_name,crop.shape[1],crop.shape[0],new_boxes,count]
                f= open(ann_path,'r')
                updatexml(ann_path,new) 
                
                
                cv2.imshow('crop_resize', crop_resize)
                cv2.imshow('resize', resize)
                cv2.waitKey (0)
                
        #cv2.imshow('origin', img)
        
                 

if __name__ == '__main__':
    preprocess()