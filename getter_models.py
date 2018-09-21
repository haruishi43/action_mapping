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
                        class_ids = test_class_ids)
        chainer.serializers.load_npz(modelfile, model)

        # use gpu
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
        bn_to_affine(model)

        return model

    def predict(self, img):
        _bboxes, _labels, _scores, _masks = self.model.predict([img])
        return _bboxes[0], _labels[0], _scores[0], _masks[0]


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
        return self.model(img)  # returns poses, scores
