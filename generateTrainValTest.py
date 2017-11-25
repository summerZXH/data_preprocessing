#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
###########
Generate train, val, test index file.

Usage:
python generateTrainValTest.py src.txt dst.txt

'''

import random
import os

def main(argv):
    assert(len(argv) == 3)
    
    # load all index
    with open(argv[1], 'r') as fid:
        indexes = [x.split('\n')[0] for x in fid.readlines()]

    train_file = os.path.join(argv[2],"scene_extra_train.txt")
    val_file = os.path.join(argv[2],"scene_extra_val.txt")
    test_file = os.path.join(argv[2],"scene_extra_test.txt")

    
    train_ratio = 0.999*0.8
    val_ratio = 0.999*0.2
    test_ratio = 0.001

    images_num=len(indexes)
    index=random.sample(set(range(0,images_num)), images_num)

    train_index = index[0 : int(images_num*train_ratio)]
    val_index = index[int(images_num*train_ratio) : int(images_num*(train_ratio + val_ratio))]
    test_index = index[int(images_num*(train_ratio + val_ratio)) :  images_num]

    with open(train_file, 'w') as fid:
        for i in train_index:
            fid.write(indexes[i] + "\n")
    with open(val_file, 'w') as fid:
        for i in val_index:
            fid.write(indexes[i] + "\n")
    with open(test_file, 'w') as fid:
        for i in test_index:
            fid.write(indexes[i] + "\n")

if __name__ == "__main__":
    print(__doc__)
    import sys
    main(sys.argv)
