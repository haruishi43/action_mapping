import os
import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import io
import json

curdir = os.path.dirname(os.path.abspath(__file__))
pardir = os.path.dirname(curdir)
sys.path.insert(0, pardir)
from pyrs import PyRS
from getter_models import MaskRCNN, OpenPose, coco_label_names, JointType, params


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def draw_person_pose(self, orig_img, poses):
    if len(poses) == 0:
        return orig_img

    limb_colors = [
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
        [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
        [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
        [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
    ]

    joint_colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    canvas = orig_img.copy()

    # limbs
    for pose in poses.round().astype('i'):
        for i, (limb, color) in enumerate(zip(params['limbs_point'], limb_colors)):
            if i != 9 and i != 13:  # don't show ear-shoulder connection
                limb_ind = np.array(limb)
                if np.all(pose[limb_ind][:, 2] != 0):
                    joint1, joint2 = pose[limb_ind][:, :2]
                    cv2.line(canvas, tuple(joint1), tuple(joint2), color, 2)

    # joints
    for pose in poses.round().astype('i'):
        for i, ((x, y, v), color) in enumerate(zip(pose, joint_colors)):
            if v != 0:
                cv2.circle(canvas, (x, y), 3, color, -1)
    return canvas

def process_all(self, color_image, depths_image, bboxes, labels, scores, masks):
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


if __name__ == '__main__':
    cap=cv2.VideoCapture(1)
    clientsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect(('localhost', 8089))

    while(cap.isOpened()):
        ret,frame = cap.read()
        data = json.dumps(frame, cls=MyEncoder).encode('utf-8')

        clientsocket.sendall(data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()