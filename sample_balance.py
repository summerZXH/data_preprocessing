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

annopath="/home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/Annotations_origin"
imgpath="/home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/JPEGImages"
new_annopath="/home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/Annotations"
scale = 1

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
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


def updatexml(ann_path):
    updatetree=ET.parse(ann_path)
    root=updatetree.getroot()
    filename=root.find('filename').text
    print('*********************************')
    for obj in updatetree.findall('object'):
        name = str(obj.find('name').text)
        if(name=='nest' or name=='text'):
            print("delete ------")
            root.remove(obj)
    updatetree.write(os.path.join(new_annopath,str(filename)+'.xml'))

def preprocess():
    #import ipdb
    #ipdb.set_trace()

    filelist=os.listdir(annopath)
    for file in filelist:
        # delete nest and text
        ann_path = os.path.join(annopath,file)
        updatexml(ann_path)

        new_ann_path = os.path.join(new_annopath,file)
        ann = parse_rec(new_ann_path)

        name = os.path.splitext(file)[0]
        jpg = name + '.jpg'
        jpg_path = os.path.join(imgpath,jpg)
        img=cv2.imread(jpg_path)
   
        width = img.shape[1]
        height = img.shape[0]
        
        boxes = []
        for i in range(len(ann)):
            x1 = ann[i]['bbox'][0]
            y1 = ann[i]['bbox'][1]
            x2 = ann[i]['bbox'][2]
            y2 = ann[i]['bbox'][3]
            boxes.append([x1,y1,x2,y2])
            
        resize = cv2.resize(img, (int(width / scale), int(height / scale)))
        for index in range(len(boxes)):
            x1 = boxes[index][0]
            y1 = boxes[index][1] 
            x2 = boxes[index][2]
            y2 = boxes[index][3]
            cv2.rectangle(resize, (int(x1/scale), int(y1/scale)), (int(x2/scale), int(y2/scale)), (255, 255, 0), 2)  
        
        #cv2.imshow('resize', resize)
        #cv2.waitKey (0)

    print("delete text and nest sucessful ...")                              

if __name__ == '__main__':
    preprocess()