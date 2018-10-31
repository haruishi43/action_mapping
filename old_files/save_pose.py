import argparse
import sys
import os
from datetime import datetime as datetime

import numpy as npd
import matplotlib.pyplot as plot
from datetime import datetime as dt

from utils import DataManagement
from save_object_pose import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= '/mnt/extHDD/raw_data',help='relative data path from where you use this program')
    parser.add_argument('--save', default= './data',help='relative saving directory from where you use this program')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    print('Getting data from: {}'.format(args.data))
    print('Saving to: {}'.format(args.save))
    dm = DataManagement(args.data, args.save)
    after = dt(2018, 9, 9, 13, 7, 0)
    before = dt(2018, 9, 9, 13, 8, 0)
    datetimes = dm.get_datetimes_in(after, before)

    # camera params
    o3_chain = Open3D_Chain()
    K = o3_chain.get_K()
    P = o3_chain.get_P()

    # Intialize Models:
    openpose = OpenPose(0)

    for datetime in datetimes:
        datetime_save_dir = dm.get_save_directory(datetime)

        if not dm.check_path_exists(datetime_save_dir):
            print('Making a save directory in: {}'.format(datetime_save_dir))
            os.makedirs(datetime_save_dir)
        else:
            print("Directory exists")


        rgb_path = dm.get_rgb_path(datetime)
        depth_path = dm.get_depth_path(datetime)

        # sort rgb files before looping
        # order matters!
        filenames = dm.get_sorted_rgb_images(datetime)
    
        # loop
        for fn in filenames:
            if fn.endswith(".png"): 
                print('\nimage: ', fn)
                filename = fn.split('.')[0] + '.npz'
                file_save_path = os.path.join(datetime_save_dir, filename)

                if dm.check_path_exists(file_save_path):
                    
                    #TODO: open the file and see if poses are in it

                    continue
                    
                # find the corresponding depth image
                rgb_img = os.path.join(rgb_path, fn)
                depth_img = os.path.join(depth_path, fn)
                if not os.path.exists(depth_img):
                    print('Could not find corresponding depth image in: {}'.format(depth_img))
                    continue

                o3_chain.change_image(rgb_img, depth_img)
                rgb = o3_chain.get_rgb()
                depths = o3_chain.get_depths()

                # OpenPose
                poses, scores = openpose.predict(rgb)
                dict_poses = get_pose(depths, K, P, poses, scores)

                # save the files as npz
                if not len(dict_poses):
                    continue
                else:
                    np.savez_compressed(file_save_path, dict_poses=dict_poses)
                    print("saved")
    