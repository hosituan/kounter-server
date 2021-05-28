# import keras
import keras

# import keras_retinanet
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from object_detector_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from object_detector_retinanet.keras_retinanet.utils.colors import label_color
# # import for EM Merger and viz
from object_detector_retinanet.keras_retinanet.utils import EmMerger
# from utils import create_folder, root_dir
import sys
# import miscellaneous modules
import cv2
import os
import numpy as np
import time
import math 

from globalModel import GlobalModel

from tensorflow.python.keras.backend import get_session
import keras
import tensorflow as tf

def distance(x1, y1, x2, y2):
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) * 1.0)

def startCount(filePath):
    graph = GlobalModel.graph
    image_path = filePath 
    image = cv2.imread(image_path)
    # for filtering predictions based on score (objectness/confidence)
    threshold = 0.3
    # copy to draw on
    draw = image.copy()
    # preprocess image for network
    image = preprocess_image(image)
    #image, scale = resize_image(image)

    # Run inference
    with graph.as_default():
        boxes, hard_scores, labels, soft_scores = GlobalModel.model.predict_on_batch(np.expand_dims(image, axis=0))

        hard_score_rate=.3
        max_detections = 9999
        soft_scores[:, :, 0]
        # soft_scores = np.squeeze(soft_scores, axis=-1)
        soft_scores = hard_score_rate * hard_scores + (1 - hard_score_rate) * soft_scores

        # correct boxes for image scale
        #boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(hard_scores[0, :] > threshold)[0]

        # select those scores
        scores = soft_scores[0][indices]
        hard_scores = hard_scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_scores = scores[scores_sort]
        image_hard_scores = hard_scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        results = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
            np.expand_dims(image_labels, axis=1)], axis=1)
        filtered_data = EmMerger.merge_detections(image_path, results)
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        csv_data_lst = []
        csv_data_lst.append(['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'hard_score'])

        for ind, detection in filtered_data.iterrows():
            box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])
            filtered_boxes.append(box)
            filtered_scores.append(detection['confidence'])
            filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
            row = [image_path, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                    detection['confidence'], detection['hard_score']]
            csv_data_lst.append(row)


        print("Count result: " + str(len(filtered_boxes)))
        count = len(filtered_boxes)  
        count_temp = 0
        dict_res = []
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            if score < threshold:
                break
            b = box.astype(int)
            dict_result = {}
            dict_result['x'] = int(b[0])
            dict_result['y'] = int(b[1])

            height = distance(b[0], b[1], b[0], b[3]) # (x1, y1) (x1, y2)
            width = distance(b[0], b[1], b[2], b[1])  
            dict_result['height'] = int(height)
            dict_result['width'] = int(width)
            dict_result["score"] = round(score, 2)
            dict_res.append(dict_result)
        return dict_res  

