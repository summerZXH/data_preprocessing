#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import os
import random

def read_txt(txt_file):
    image_list=[] 
    label_list=[]
    numbers=0
    
    with open(txt_file) as f:
        def_lines = f.readlines()
    def_lines.pop(0)
    
    #import ipdb
    #ipdb.set_trace()

    for def_line in def_lines:   
        print(def_line)
        if (def_line.find('jpg')>0):
            image_tmp = def_line.split('jpg')[0]
            image_tmp = image_tmp + 'jpg'
            image_list.append(image_tmp)
            label_tmp = int(def_line.split('jpg')[1])
            label_list.append(label_tmp)
        else:
            image_tmp = def_line.split('JPEG')[0]
            image_tmp = image_tmp + 'JPEG'
            image_list.append(image_tmp)
            label_tmp = int(def_line.split('JPEG')[1])
            label_list.append(label_tmp)

    numbers = len(image_list)
      
    return image_list, label_list, numbers
def read_json(json_file):
    with open(json_file) as f:
        anno = json.load(f)  
    num=len(anno)
    index=random.sample(set(range(0,num)), num)
    
    ratio = 0.7
    train_index = index[0 : int(num*ratio)]
    val_index = index[int(num*ratio) : num]
    
    train_file = 'scene_val_train.json'
    val_file = 'scene_val_test.json'

    with open(train_file, 'w') as fid:
        train_result = []
        for i in train_index:
            train_result.append(anno[i])
        json.dump(train_result, fid)
        
    with open(val_file, 'w') as fid:
        val_result = []
        for i in val_index:
            val_result.append(anno[i])
        json.dump(val_result, fid)
    print('sucessful...')
    print(num, len(train_result), len(val_result))

def write_json(json_file, image_list, label_list):
    result = []
    for image, label in zip(image_list, label_list):
        temp_dict = {}
        temp_dict['image_id'] = image       
        temp_dict['label_id'] = label
        result.append(temp_dict)

    with open(json_file, 'w') as f:
        json.dump(result, f)
   
    print('write result json, num is %d' % 
            len(result))

def main(args):
    read_json(args.val_json_file)
    #image_list, label_list, numbers = read_txt(args.txt_file)
    #write_json(args.json_file, image_list, label_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert txt to json')
    # addarg here
    parser.add_argument('--txt_file', default='train_val_test_txt/scene_extra_test.txt', type=str, help='txt file')
    parser.add_argument('--json_file', default='train_val_test_txt/scene_extra_test.json', type=str, help='json file')   
    parser.add_argument('--data_base', default='/home/zhuxiuhong/src/ai_challenge/scene/Extra/train', type=str, help='data base') 
    parser.add_argument('--val_json_file', default='./scene_validation_annotations_20170908.json', type=str, help='json file')   
    args = parser.parse_args()
    main(args)



