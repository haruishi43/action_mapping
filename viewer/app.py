import os
import sys
import time

import numpy as np
from flask import Flask, render_template, Response, jsonify, json
from flask_socketio import SocketIO

sys.path.append('..')
from utils import poses_objects_from_npz

dir_path = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)


##########################################################################################################################
#
# Pose Getter (an iterator)
#
##########################################################################################################################

class PoseObjectGetter:
    """Pose iterator"""
    def __init__(self, file_dir='../data/20180913_1908'):
        self.data_path = os.path.join(dir_path, file_dir)
        files = os.listdir(self.data_path)
        self.filenames = sorted(files, key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.current = 0
        self.total_files = len(files)

    def __iter__(self):
        '''Not really using it'''
        return self

    def __next__(self):
        if self.current == len(self.filenames):
            # reset
            self.current = 0
        
        file_path = os.path.join(self.data_path, self.filenames[self.current])
        poses, bboxes, centers = poses_objects_from_npz(file_path)

        self.current += 1

        poses = self._to_json(poses)
        bboxes = self._bboxes_to_json(bboxes)
        centers = self._centers_to_json(centers)

        return poses, bboxes, centers

    def _cleanse_dict(self, _in):
        _out = {}

        if _in is not None:  # send empty dict if None
            ids = list(_in.keys())
        
            for id in ids:
                single = _in[id]  # np array
                single = 0.01 * single  # change scale

                # remove nans and make it into a dict of lists
                single_dict = {i: s.tolist() for i, s in enumerate(single) if not np.isnan(s).any()}
                _out[id] = single_dict  # add to dict
        
        return _out

    def _to_json(self, _in):

        # get cleansed dictionary
        _in = self._cleanse_dict(_in)

        # convert to json
        _out = json.dumps(_in)
        return _out

    def _bboxes_to_json(self, _in):
        _out = {}

        if _in is not None:  # send empty dict if None
            ids = list(_in.keys())
        
            for id in ids:
                single = _in[id]  # np array
                single = 0.01 * single  # change scale

                # remove nans and make it into a dict of lists
                single_dict = single.tolist()
                _out[id] = single_dict  # add to dict

        # convert to json
        _out = json.dumps(_out)
        return _out

    def _centers_to_json(self, _in):
        _out = {}

        if _in is not None:  # send empty dict if None
            ids = list(_in.keys())
        
            for id in ids:
                single = _in[id]  # np array
                single = 0.01 * single  # change scale

                # remove nans and make it into a dict of lists
                single_dict = single.tolist()
                _out[id] = single_dict  # add to dict

        # convert to json
        _out = json.dumps(_out)
        return _out



getter = PoseObjectGetter(file_dir='../data/20180909_1310')


##########################################################################################################################
#
#  socketio / flask
#
##########################################################################################################################


def get_data_from_getter():
    pose, bbox, center = next(getter)
    return (pose, bbox, center)


def messageReceived(methods=['GET', 'POST']):
    print('message received!')


def ack():
    print('message received!')


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('give me data')
def handle_pose(_json, methods=['GET', 'POST']):
    json_data = get_data_from_getter()
    return json_data


@socketio.on('ack connection')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    socketio.emit('my response', json, callback=messageReceived)


if __name__ == '__main__':
    # socketio.run(app, debug=True)
    socketio.run(app, host='localhost', port=5050, debug=True)

    ## Testing getter:
    from pprint import pprint
    
    # for ps, bbox in getter:
    #     print(bbox)

    # pose, bbox, center = next(getter)
    # pprint(pose)

    # print('')
    # pprint(bbox)

    # print('')
    # pprint(center)
