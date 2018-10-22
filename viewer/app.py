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


class PoseGetter:
    """Pose iterator"""
    def __init__(self, file_dir='../data/20180913_1908'):
        self.data_path = os.path.join(dir_path, file_dir)
        files = os.listdir(self.data_path)
        self.filenames = sorted(files, key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.current = 0

    def __iter__(self):
        '''Not really using it'''
        return self

    def __next__(self):
        if self.current == len(self.filenames):
            # reset
            self.current = 0
        
        file_path = os.path.join(self.data_path, self.filenames[self.current])
        poses, _, _ = poses_objects_from_npz(file_path)

        self.current += 1

        return self._to_json(poses)

    def _cleanse_data(self, _poses):
        poses = {}

        if _poses is not None:  # send empty dict if no poses
            pose_ids = list(_poses.keys())
        
            for p_id in pose_ids:
                pose = _poses[p_id]  # np array
                pose = 0.01 * pose  # change scale

                # remove nans and make it into a dict of lists
                pose_dict = {i: pos.tolist() for i, pos in enumerate(pose) if not np.isnan(pos).any()}
                poses[p_id] = pose_dict  # add to dict
        
        return poses

    def _to_json(self, _poses):

        # get cleansed dictionary
        _poses = self._cleanse_data(_poses)

        # convert to json
        poses = json.dumps(_poses)
        return poses


pose_getter = PoseGetter()



def get_pose():
    pose = next(pose_getter)
    return pose


def messageReceived(methods=['GET', 'POST']):
    print('message received!')

def ack():
    print('message received!')

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('want pose')
def handle_pose(_json, methods=['GET', 'POST']):
    json_data = get_pose()
    return json_data
    # socketio.emit('send pose', json_data, callback=ack)


@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    socketio.emit('my response', json, callback=messageReceived)


if __name__ == '__main__':
    socketio.run(app, debug=True)

    # from pprint import pprint
    # for ps in pose_getter:

    #     pprint(ps)