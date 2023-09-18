# YOLO object detection
import os
import cv2 as cv
import numpy as np
import time

MIN_CONF = 0.5

def extract_tags(image_path: str):
    """
    Extract tags from an image.
    :param image_path: path to image
    :return: list of tags
    """
    p = os.path.join("uploads/",image_path)
    h,w=416,416
    img = cv.imread(p)
    
    classes = open('tagger_files/coco.names').read().strip().split('\n')

    # Give the configuration and weight files for the model and load the network.
    net = cv.dnn.readNetFromDarknet('tagger_files/yolov3.cfg', 'tagger_files/yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(img, 1/255.0, (h, w), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    net.setInput(blob)
    outputs = net.forward(ln)

    # boxes = []
    confidences = []
    classIDs = []
    # h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > MIN_CONF:
                # box = detection[:4] * np.array([w, h, w, h])
                # (centerX, centerY, width, height) = box.astype("int")
                # x = int(centerX - (width / 2))
                # y = int(centerY - (height / 2))
                # box = [x, y, int(width), int(height)]
                # boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classes[classID])

    # for confidence, clsID in zip(confidences, classIDs):
    #     print(round(confidence, 3), clsID)

    return list(set(classIDs))