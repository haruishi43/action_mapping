import os, sys
import cv2
import numpy as np
from base_camera import BaseCamera

curdir = os.path.dirname(os.path.abspath(__file__))
pardir = os.path.dirname(curdir)
sys.path.insert(0, pardir)
from pyrs import PyRS
from getter_models import MaskRCNN, OpenPose, coco_label_names, JointType, params

class Camera(BaseCamera):

    def draw_person_pose(orig_img, poses):
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

    @staticmethod
    def frames():
        # initialize models

        maskrcnn = MaskRCNN(0)
        openpose = OpenPose(0)

        w = 1280
        h = 720

        t = 0
        with PyRS(h=720, w=1280) as pyrs:
            while True:
                if t % 4 == 0:
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

                    if len(labels) > 0:
                        # all_depths = process_one(color_image, depths_image, bbox, label, score, mask, item_index)
                        all_depths = Camera.process_all(color_image, depths_image, bboxes, labels, scores, masks)
                    else:
                        all_depths = np.zeros([h, w])

                    # items_image = cv2.cvtColor(all_depths.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    # images = np.hstack((color_image, items_image))
                    yield cv2.imencode('.jpg', color_image)[1].tobytes(), t
                    t = 0
                t += 1