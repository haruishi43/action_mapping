import os, sys
import cv2
import numpy as np
from base_camera import BaseCamera

curdir = os.path.dirname(os.path.abspath(__file__))
pardir = os.path.dirname(curdir)
sys.path.insert(0, pardir)
from pyrs import PyRS
from object_trackering_demo import *

class Camera(BaseCamera):
    
    @staticmethod
    def frames():
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

        with PyRS(h=720, w=1280) as pyrs:
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
                yield cv2.imencode('.jpg', images)[1].tobytes()
