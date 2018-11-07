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

from utils import coco_label_ids


MASKRCNN_MODEL_FILE = 'modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz'
OPENPOSE_MODEL_FILE = 'models/coco_posenet.npz'

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
