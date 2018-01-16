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

annopath="/home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/Annotations"
imgpath="/home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/JPEGImages"
new_xml_dir="/home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/Annotations1"
new_img_dir="/home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/JPEGImages1"

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

def updatexml(ann_path,new):
    updatetree=ET.parse(ann_path)
    print ann_path
    root=updatetree.getroot()
    filename=root.find('filename')
    filename.text=str(new)
    updatetree.write(os.path.join(new_xml_dir,new+'.xml'))

def preprocess():
    filelist=os.listdir(annopath)
    num = 1
    number = 0
    for file in filelist:
        print('number =  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',number)
        number += 1
        flag = False
        ann_path = os.path.join(annopath,file)
        ann = parse_rec(ann_path)
        name = os.path.splitext(file)[0]
        jpg = name + '.jpg'
        jpg_path = os.path.join(imgpath,jpg)
        img=cv2.imread(jpg_path)
        new_name = (6 - len(str(num)))*'0' + str(num)
        num+=1
        cv2.imwrite(os.path.join(new_img_dir,new_name+'.jpg'), img)
        updatexml(ann_path,new_name) 
        #import ipdb
        #ipdb.set_trace()

if __name__ == '__main__':
    preprocess()
