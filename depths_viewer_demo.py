import sys
import os

import numpy as np
import cv2

from pyrs import PyRS
from getter_models import MaskRCNN
from utils import object_dict

object_ids = object_dict.keys()


def process_one(color_image, depths_image, bboxes, labels, scores, masks, item_index):
    all_depths = np.zeros(depths_image.shape)
    items = np.where(labels == item_index)[0]
    if items.any():
        for item in items:
            name = object_dict[item_index]
            item_mask = masks[item]
            item_depth = np.multiply(depths_image, item_mask)
            all_depths = np.add(all_depths, item_depth)
            # beautify:
            y1, x1, y2, x2 = [int(n) for n in bboxes[item]]
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(color_image, name, (x1 + 10, y1 + 10), 0, 0.3, (0,255,0))
        
        all_depths *= 255 / all_depths.max()

    return all_depths
    

def process_all(color_image, depths_image, bboxes, labels, scores, masks):
    all_depths = np.zeros(depths_image.shape)

    if len(labels):
        for i, label in enumerate(labels):
            
            if label in object_dict:
                name = object_dict[label]
                item_mask = masks[i]
                item_depth = np.multiply(depths_image, item_mask)
                all_depths = np.add(all_depths, item_depth)
                # beautify:
                y1, x1, y2, x2 = [int(n) for n in bboxes[i]]
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(color_image, name, (x1 + 10, y1 + 10), 0, 0.3, (0,255,0))

        all_depths *= 255 / all_depths.max()

    return all_depths


if __name__ == '__main__':

    maskrcnn = MaskRCNN()

    w = 1280
    h = 720

    item_index = 1
    name = object_dict[item_index]
    
    with PyRS(w=w, h=h) as pyrs:
        print('Modes:')
        print('\tExit:\tq')

        preset = pyrs.get_depths_preset()
        preset_name = pyrs.get_depths_preset_name(preset)
        print('Preset: ', pyrs.get_depths_preset_name(preset))

        while True:
            # Wait for a coherent pair of frames: depth and color
            pyrs.update_frames()

            # Get images as numpy arrays
            color_image = pyrs.get_color_image()
            depths_image = pyrs.get_depths_frame()
            color = color_image.swapaxes(2, 1).swapaxes(1, 0)
            
            bbox, label, score, mask = maskrcnn.predict(color)
            if label is not None:
                # all_depths = process_one(color_image, depths_image, bbox, label, score, mask, item_index)
                all_depths = process_all(color_image, depths_image, bbox, label, score, mask)
            else:
                all_depths = np.zeros([h, w])

            items_image = cv2.cvtColor(all_depths.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            images = np.hstack((color_image, items_image))
            
            # Show image
            cv2.namedWindow('detection', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('detection', images)
            key = cv2.waitKey(10)

            if key == ord('q'):
                # end OpenCV loop
                break
            elif key == ord('p'):
                # save rgb and depths
                cv2.imwrite("static_data/detection_results_color.png".format(name), color_image)
                cv2.imwrite("static_data/detection_results_depth.png".format(name), depths_image)
                cv2.imwrite("static_data/detection_results.png", images)