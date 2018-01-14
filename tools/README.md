## step0: init
    ```
    $ python createDS.py
    ```
## step1: check and delete bad or empty xmls 
    ```
    $ python checkAnno.py --anno_dir /home/zhuxiuhong/src/project/object_detection/1_faster_rcnn/data/VOCdevkit2007/VOC2007/Annotations
    ```
## step2: rename xmls
    ```
    $ python rename.py
    ```
## step3: split to train/val/test
    ```
    $ python createSplit.py --cfg_file /home/zhuxiuhong/src/project/object_detection/BBox-Label-Tool/sample.bbox_label.txt
    ```