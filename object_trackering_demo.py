import sys
import os

import chainer
import numpy as np
import open3d as o3
import cv2
from pyrs import PyRS
import matplotlib.pyplot as plot
from chainercv import utils

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_maskrcnn = os.path.join(dir_path, 'maskrcnn')
from maskrcnn import MaskRCNNTrainChain
from maskrcnn import freeze_bn, bn_to_affine
from maskrcnn import MaskRCNNResNet
from maskrcnn import vis_bbox


def process_one(color_image, depths_image, bboxes, labels, scores, masks, item_index):
    all_depths = np.zeros(depths_image.shape)
    items = np.where(labels == item_index)[0]
    if items.any():
        for item in items:
            name = coco_label_names[item_index]
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
            name = coco_label_names[label]
            item_mask = masks[i]
            item_depth = np.multiply(depths_image, item_mask)
            all_depths = np.add(all_depths, item_depth)
            # beautify:
            y1, x1, y2, x2 = [int(n) for n in bboxes[i]]
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(color_image, name, (x1 + 10, y1 + 10), 0, 0.3, (0,255,0))

        all_depths *= 255 / all_depths.max()

    return all_depths


test_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, \
    57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

coco_label_names = ('background',  # class zero
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'mirror', 'dining table', 'window', 'desk','toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

if __name__ == '__main__':

    roi_size = 14
    modelfile = os.path.join(abs_maskrcnn, 'modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz')
    roi_align = True

    model = MaskRCNNResNet(n_fg_class = 80,
                        roi_size = roi_size,
                        pretrained_model = modelfile,
                        n_layers = 50,  # resnet 50 layers (not 101 layers)
                        roi_align = roi_align,
                        class_ids = test_class_ids)
    chainer.serializers.load_npz(modelfile, model)

    chainer.cuda.get_device_from_id(0).use()
    model.to_gpu()
    bn_to_affine(model)

    w = 1280
    h = 720
    name = 'chair'
    item_index = coco_label_names.index(name)
    
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
            
            bboxes, labels, scores, masks = model.predict([color])
            if len(labels) > 0:
                bbox, label, score, mask = bboxes[0], np.asarray(labels[0], dtype=np.int32), scores[0], masks[0]
                # all_depths = process_one(color_image, depths_image, bbox, label, score, mask, item_index)
                all_depths = process_all(color_image, depths_image, bbox, label, score, mask)
            else:
                all_depths = np.zeros([h, w])

            items_image = cv2.cvtColor(all_depths.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            images = np.hstack((color_image, items_image))
            
            # Show image
            cv2.namedWindow('Item Tracker', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Item Tracker', images)
            key = cv2.waitKey(10)

            if key == ord('q'):
                # end OpenCV loop
                break
            elif key == ord('p'):
                # save rgb and depths
                cv2.imwrite("static_data/color_{}.png".format(name), color_image)
                cv2.imwrite("static_data/depth_{}.png".format(name), depths_image)
                cv2.imwrite("static_data/tracker.png", images)