#############################################
# Object detection by opencv
# Function: Single or multiple test
# Update : Sam Su
# Date: December 2, 2021
#############################################

import cv2
import numpy as np
import os
import glob
import time


def get_basename(filepath):
    '''
        filename and extension name
        filepath = '/home/ubuntu/python/example.py'
        return basename = example.py
    '''
    return os.path.basename(filepath)

def get_filename_only(filepath):
    '''filepath = '/home/ubuntu/python/example.py'
       return filename = example
    '''
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]  

def created_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def get_obj(imgpath, net):
    '''main function for get bbox'''
    # print('Image:{}'.format(imgpath))
    yolonet = yolo(net)
    srcimg = cv2.imread(imgpath)
    nms_dets, frame = yolonet.detect(srcimg)
    return nms_dets, frame

def single(f, output_dir, net):
    dets, frame = get_obj(f, net)
    print(dets)
    output_file_name = os.path.join(output_dir, get_basename(f))
    cv2.imwrite(output_file_name, frame)
    print('Save the result to:{}'.format(output_file_name))
    
def multi(path, output_dir, net):
    for f in glob.glob(os.path.join(path, "*.jpg")):
        single(f, output_dir, net)

def testN(f, output_dir, net, N):
    for i in range(N):
        tStart = time.time()
        single(f, output_dir, net)
    print('{} Spend time:{}'.format(i, time.time()-tStart))


class yolo():
    def __init__(self, net):
        # print('Net use', config['netname'])
        self.confThreshold = net['confThreshold']
        self.nmsThreshold = net['nmsThreshold']
        # print('__init__.self.nmsThreshold', self.nmsThreshold)

        self.inpWidth = net['inpWidth']
        self.inpHeight = net['inpHeight']
        with open(net['classesFile'], 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        self.net = cv2.dnn.readNet(net['modelConfiguration'], net['modelWeights'])

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        nms_classIds = []
        nms_confidences = []
        nms_boxes = []
        nms_dets = []
        # print('postprocess.self.nmsThreshold', self.nmsThreshold)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
            nms_classIds.append(classIds[i])
            label = str(self.classes[classIds[i]])
            nms_confidences.append(confidences[i])
            nms_boxes.append([left, top, width, height])
            nms_dets.append([label, confidences[i], left, top, left + width, top + height, classIds[i]])
        return nms_dets, frame

    def detect(self, srcimg):
        # print('detect.self.nmsThreshold', self.nmsThreshold)
        blob = cv2.dnn.blobFromImage(srcimg, 1/255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        nms_dets, frame = self.postprocess(srcimg, outs)
        return nms_dets, frame


# =============================================================================
# The following main functions are used for standalong testing
# =============================================================================
if __name__ == "__main__":
    yolo_config = './yolov4-tiny.cfg'
    yolo_model = './yolov4-tiny.weights'
    net = {
        'confThreshold':0.3, 
        'nmsThreshold':0.3, 
        'inpWidth':512, 
        'inpHeight':512, 
        'classesFile':'coco.names', 
        'modelConfiguration': yolo_config, 
        'modelWeights': yolo_model, 
        'netname':'yolov4-tiny'}

    output_dir = './output'
    created_directory(output_dir)

# =============================================================================
#     # === for single ===
# =============================================================================
    img_file = 'bus.jpg'
    single(img_file, output_dir, net)

# =============================================================================
#     # === for multiple ===
# =============================================================================
#    imput_dir = 'dir'
#    multi(imput_dir, output_dir, net)

# =============================================================================
#     # ===for test===
# =============================================================================
#    N = 10
#    img_file = 'bus.jpg'
#    testN(img_file, output_dir, net, N)
