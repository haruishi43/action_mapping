import os, sys
import cv2
import numpy as np
from base_camera import BaseCamera

curdir = os.path.dirname(os.path.abspath(__file__))
pardir = os.path.dirname(curdir)
sys.path.insert(0, pardir)
from pyrs import PyRS
from getter_models import MaskRCNN, OpenPose, coco_label_names, JointType, params
from save_object_pose import *

class Camera(BaseCamera):

    @staticmethod
    def data():
        # initialize models

        maskrcnn = MaskRCNN(0)
        openpose = OpenPose(0)

        w = 1280
        h = 720

        o3_chain = Open3D_Chain()
        K = o3_chain.get_K()
        P = o3_chain.get_P()

        with PyRS(h=720, w=1280) as pyrs:

            while True:
                # Wait for a coherent pair of frames: depth and color
                pyrs.update_frames()

                # Get images as numpy arrays
                color_image = pyrs.get_color_image()
                depths_image = pyrs.get_depths_frame()
                color = color_image.swapaxes(2, 1).swapaxes(1, 0)
                
                # predictions
                bboxes, labels, scores, masks = maskrcnn.predict(color)
                poses, _ = openpose.predict(color_image)

                color_image = Camera.draw_person_pose(color_image, poses)

                
                yield 
                