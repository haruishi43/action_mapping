import argparse
import sys
import os
from datetime import datetime as dt

import numpy as np
import chainer

from open3d_chain import Open3D_Chain
from utils import DataManagement

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_op_lib = os.path.join(dir_path, 'openpose')
from openpose import params, JointType
from openpose import PoseDetector, draw_person_pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= '/mnt/extHDD/raw_data',help='relative data path from where you use this program')
    parser.add_argument('--save', default= 'pose',help='relative saving directory from where you use this program')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # set up for Chainer
    chainer.config.enable_backprop = False
    chainer.config.train = False

    # get directory of data (rgb, depth)
    print('Getting data from: {}'.format(args.data))
    dm = DataManagement(args.data)
    after = dt(2018, 9, 9, 0, 0, 0)
    before = dt(2018, 9, 10, 0, 0, 0)
    datetimes = dm.get_datetimes_in(after, before)

    # camera params
    o3_chain = Open3D_Chain()

    # load model
    pose_detector = PoseDetector("posenet", 
                                os.path.join(abs_op_lib, 
                                "models/coco_posenet.npz"), 
                                device=args.gpu)

    for datetime in datetimes:
        save_path = dm.get_save_directory(datetime)
        save_path = os.path.join(save_path, args.save)
        if not dm.check_path_exists(save_path):
            print('Making a save directory in: {}'.format(save_path))
            os.makedirs(save_path)
        else:
            continue

        rgb_path = dm.get_rgb_path(datetime)
        depth_path = dm.get_depth_path(datetime)

        # sort rgb files before looping
        # order matters!
        filenames = dm.get_sorted_rgb_images(datetime)
    
        # Loop:
        pose_num = 0
        for fn in filenames:
            if fn.endswith(".png"): 
                print('\nimage: ', fn)
                # find the corresponding depth image
                rgb_img = os.path.join(rgb_path, fn)
                depth_img = os.path.join(depth_path, fn)
                if not os.path.exists(depth_img):
                    print('Could not find corresponding depth image in: {}'.format(depth_img))
                    continue

                # read image
                o3_chain.change_image(rgb_img, depth_img)

                # inference
                poses, scores = pose_detector(o3_chain.get_rgb())

                for i, pose in enumerate(poses):
                    csv_name = str(pose_num) + '.csv'

                    joints = np.zeros((len(JointType), 3))

                    for i, joint in enumerate(pose):
                        x, y = int(joint[0]), int(joint[1])
                        Z = o3_chain.get_depths()[y][x]

                        if Z != 0.0:
                            # Depth = 0 means that depth data was unavailable
                            X, Y = o3_chain.calc_xy(x, y, Z)
                            joints[i] = np.asarray([X, Y, Z])
                            # print('x: {}, y: {}, depth: {}'.format(X, Y, Z))
                    
                    csv_path = os.path.join(save_path, csv_name)
                    np.savetxt(csv_path, joints, delimiter=",")
                    
                    pose_num += 1
    