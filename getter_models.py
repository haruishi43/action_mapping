import os

import chainer

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_maskrcnn = os.path.join(dir_path, 'maskrcnn')
from maskrcnn import MaskRCNNTrainChain
from maskrcnn import freeze_bn, bn_to_affine
from maskrcnn import MaskRCNNResNet
from maskrcnn import vis_bbox

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_op_lib = os.path.join(dir_path, 'openpose')
from openpose import params, JointType
from openpose import PoseDetector, draw_person_pose


MASKRCNN_MODEL_FILE = 'modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz'
OPENPOSE_MODEL_FILE = 'models/coco_posenet.npz'


# len = 41
extracting_ids = [1, \
    26, 27, 30, \
    31, 33, \
    44, 45, \
    46, 47, 48, 49, 50, \
    51, 52, 53, 54, 55, \
    56, 57, 58, 59, 60, \
    61, 62, 63, 64, 65, \
    67, 68, 69, \
    73, 74, \
    76, 77, 78, \
    81, 82, 84, 85, \
    87]


# 80
coco_label_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, \
    57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


coco_label_names = ('background',  # class zero
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 
    'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed',
    'mirror', 'dining table', 'window', 'desk','toilet', 
    'door', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
    'sink', 'refrigerator', 'blender', 'book', 'clock', 
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

coco_label_colors = {
    1: [1, 0.7, 0],
    2: [0, 0, 0],
    3: [0, 0, 0],
    4: [0, 0, 0],
    5: [0, 0, 0],
    6: [0, 0, 0],
    7: [0, 0, 0],
    8: [0, 0, 0],
    9: [0, 0, 0],
    10: [0, 0, 0],
    11: [0, 0, 0],
    13: [0, 0, 0],
    14: [0, 0, 0],
    15: [0, 0, 0],
    16: [0, 0, 0],
    17: [0, 0, 0],
    18: [0, 0, 0],
    19: [0, 0, 0],
    20: [0, 0, 0],
    21: [0, 0, 0],
    22: [0, 0, 0],
    23: [0, 0, 0],
    24: [0, 0, 0],
    25: [0, 0, 0],
    27: [0.5, 0, 0.5],
    28: [0.3, 0.6, 1],
    31: [0.8, 0, 0.1],
    32: [0, 0.9, 1],
    33: [0.2, 0.2, 1],
    34: [0, 0, 0],
    35: [0, 0, 0],
    36: [0.1, 0.4, 0],
    37: [0, 0, 0],
    38: [0, 0, 0],
    39: [0, 0, 0],
    40: [0, 0, 0],
    41: [0, 0, 0],
    42: [0, 0, 0],
    43: [0, 0, 0],
    44: [0.9, 0.7, 1],
    46: [0, 0.2, 0.6],
    47: [1, 0.4, 0.5],
    48: [0, 0.1, 0.5],
    49: [0.2, 1, 0.2],
    50: [0.4, 0.7, 0.7],
    51: [0, 0, 0.3],
    52: [0, 0.5, 0.1],
    53: [0.1, 0.7, 0.3],
    54: [0.6, 0.5, 0.4],
    53: [0.3, 0.2, 0.1],
    54: [0.1, 0.2, 0.3],
    55: [0.4, 0.5, 0.6],
    56: [0.9, 0.8, 0.7],
    57: [0.6, 0.7, 1],
    58: [0.1, 0.1, 0.3],
    59: [0, 1, 0.5],
    60: [0.5, 0.3, 0.8],
    61: [0.6, 0.3, 0.1],
    62: [0.1, 0.6, 0.8],
    63: [1, 0.2, 0.6],
    64: [1, 0, 0.6],
    65: [0.9, 0.1, 0.9],
    67: [0.8, 0.3, 0.8],
    70: [0.4, 0.3, 0.9],
    72: [0, 0.3, 0.3],
    73: [0, 0.7, 1],
    74: [0, 0.5, 0.5],
    75: [1, 0.3, 0.2],
    76: [0.4, 0.4, 1],
    77: [0.1, 0.4, 0.3],
    78: [0, 0, 0.5],
    79: [0.7, 0.7, 0.3],
    80: [0.4, 0.3, 0],
    81: [0.8, 0.5, 0.3],
    82: [0.6, 0.9, 0.3],
    83: [0.5, 0.6, 0.2],
    84: [0.3, 0.6, 0.6],
    85: [0.9, 0.5, 0.1],
    86: [0.3, 0.5, 0.5],
    87: [1, 1, 0.3],
    88: [0.8, 1, 1],
    89: [0.3, 0.5, 0.3],
    90: [0.7, 0.8, 0.2]
}


class GetterBase:

    def _initialize_detector(self, gpu_id):
        pass

    def predict(self, img):
        pass


class MaskRCNN(GetterBase):

    def __init__(self, gpu_id=0):
        GetterBase.__init__(self)
        self.model = self._initialize_detector(gpu_id)

    def _initialize_detector(self, gpu_id):
        assert gpu_id >= 0, "You must use a GPU"
        
        # setup chainer mask-rcnn
        roi_size = 14
        roi_align = True

        modelfile = os.path.join(abs_maskrcnn, MASKRCNN_MODEL_FILE)

        model = MaskRCNNResNet(n_fg_class = 80,
                        roi_size = roi_size,
                        pretrained_model = modelfile,
                        n_layers = 50,  # resnet 50 layers (not 101 layers)
                        roi_align = roi_align,
                        class_ids = coco_label_ids)
        chainer.serializers.load_npz(modelfile, model)

        # use gpu
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
        bn_to_affine(model)

        return model

    def predict(self, img):
        _bboxes, _labels, _scores, _masks = self.model.predict([img])
        if len(_labels[0]) > 0:
            return _bboxes[0], _labels[0], _scores[0], _masks[0]
        else: return None, None, None, None


class OpenPose(GetterBase):

    def __init__(self, gpu_id):
        GetterBase.__init__(self)
        self.model = self._initialize_detector(gpu_id)

    def _initialize_detector(self, gpu_id):
        assert gpu_id >= 0, "You must use a GPU"

        modelfile = os.path.join(abs_op_lib, OPENPOSE_MODEL_FILE)

        model = PoseDetector("posenet", 
                                    modelfile, 
                                    device=gpu_id)
        return model

    def predict(self, img):
        poses, scores = self.model(img)  # returns poses, scores
        return poses, scores
