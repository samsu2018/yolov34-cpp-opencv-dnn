# -*- coding: utf-8 -*-
#############################################
# Create a separate detection-results text file for each image
# Function: for aMP, https://github.com/samsu2018/mAP
# Update : Sam Su
# Date: December 2, 2021
#############################################

import glob
import os
from saingle_or_multi import get_obj

# classna,e
cls_file = './coco.names'
with open(cls_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def get_basename(filepath):
    '''
        filename and extension name
        filepath = '/home/ubuntu/python/example.py'
        return basename = example.py
    '''
    return os.path.basename(filepath)

def single(f, net, out_path):
    objs, frame = get_obj(f, net)
    basename = get_basename(f)
    detected_file = os.path.join(out_path, basename)
    file = open(detected_file,'w')
    for obj in objs:
        label = obj[0]
        score = obj[1]
        x1, y1, x2, y2 = obj[2], obj[3], obj[4], obj[5]

        print('{} {} {} {} {} {}'.format(label, score, x1, x2, y1, y2))
        file.write('{} {} {} {} {} {}\n'.format(label, score, x1, x2, y1, y2))
    file.close()


if __name__ == '__main__':
    ### Usagd a Benchmark for testing ...
    imagePath = 'Indoor1080tn'

    ### Parameter of YOLO model parameter
    yolo_config = './yolov4-tiny.cfg'
    yolo_model = './yolov4-tiny.weights'
    # yolo_config = './yolov4.cfg'
    # yolo_model = './yolov4.weights'

    ### test for threshold
    print('Exhaustive method')
    net = {
        'confThreshold':0.4, 
        'nmsThreshold':0.7, 
        'inpWidth':416, 
        'inpHeight':416, 
        'classesFile':'coco.names', 
        'modelConfiguration': yolo_config, 
        'modelWeights': yolo_model, 
        'netname':'yolov4-tiny'}
    
    # ---Set the output path---#
    output_diretory = './for_mAP/' + imagePath.split('/')[-1]
    if not os.path.isdir(output_diretory):  #output_directory
        os.makedirs(output_diretory)
    
    for f in glob.glob(os.path.join(imagePath, "*.jpg")):
        single(f, net, output_diretory)
