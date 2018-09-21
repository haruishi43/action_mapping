import os
import argparse
import multiprocessing as mp
from datetime import datetime as dt
from collections import defaultdict
import time

import numpy as np

from open3d_chain import Open3D_Chain
from getter_models import MaskRCNN, OpenPose, coco_label_names, JointType
from utils import DataManagement


def get_pose(openpose, q_pose, rgb_img, depth_img):
    # camera params
    o3_chain = Open3D_Chain()
    o3_chain.change_image(rgb_img, depth_img)
    rgb = o3_chain.get_rgb()
    depths = o3_chain.get_depths()
    poses, scores = openpose.predict(rgb)
    
    pose_dict = {}

    poses_num = 0
    for i, pose in enumerate(poses):

        if scores[i] < 4:  # scoring system is wierd (probably ~1 per joints?)
            continue
        
        joints = np.empty((len(JointType), 3))
        joints.fill(np.nan)

        for j, joint in enumerate(pose):
            x, y = int(joint[0]), int(joint[1])
            Z = o3_chain.get_depths()[y][x]

            if Z != 0.0:
                # Depth = 0 means that depth data was unavailable
                X, Y = o3_chain.calc_xy(x, y, Z)
                joints[j] = o3_chain.convert2world(np.asarray([X, Y, Z]))
                # print('x: {}, y: {}, depth: {}'.format(X, Y, Z))

        pose_dict[poses_num] = joints
        poses_num += 1

    q_pose.put((0, pose_dict))


def get_object(maskrcnn, q_mask, rgb_img, depth_img):
    o3_chain = Open3D_Chain()
    o3_chain.change_image(rgb_img, depth_img)
    rgb = o3_chain.get_rgb().swapaxes(2, 1).swapaxes(1, 0)
    depths = o3_chain.get_depths()
    _, labels, scores, masks = maskrcnn.predict(rgb)

    object_masks = {}
    objects = defaultdict(int)
    w = 1280  # image width

    for i, label in enumerate(labels):
        name = coco_label_names[label]
        
        if scores[i] < 0.70:
            continue

        # multiply mask with depth frame
        mask = masks[i]
        mask_with_depth = np.multiply(depths, mask)
        mask_flattened = mask_with_depth.flatten()
        # Get all indicies that has depth points
        non_zero_indicies = np.nonzero(mask_flattened)[0]

        image_size = len(depths.flatten())
        mask_size = len(non_zero_indicies)

        points = np.zeros((mask_size, 3))

        for j, index in enumerate(non_zero_indicies):
            Z = mask_flattened[index]
            
            x, y = index % w, index // w

            # get X and Y converted from pixel (x, y) using Z and intrinsic
            X, Y = o3_chain.calc_xy(x, y, Z)
            # print('x: {}, y: {}, depth: {}'.format(X, Y, Z))

            # append to points
            points[j] = o3_chain.convert2world(np.asarray([X, Y, Z]))
        
        downsampled_points = o3_chain.downsample_nparray(points, mask_size/image_size)
        
        title = name + str(objects[label])
        object_masks[title] = downsampled_points
        objects[label] += 1

    q_mask.put((1, object_masks))

    
 


if __name__ == "__main__":
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser(description='Pose Getter')
    parser.add_argument('--data', default= '/mnt/extHDD/raw_data',help='relative data path from where you use this program')
    parser.add_argument('--save', default= 'pose',help='relative saving directory from where you use this program')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    print('Getting data from: {}'.format(args.data))
    dm = DataManagement(args.data)
    after = dt(2018, 9, 9, 0, 0, 0)
    before = dt(2018, 9, 10, 0, 0, 0)
    datetimes = dm.get_datetimes_in(after, before)

    

    # Intialize Models:
    maskrcnn = MaskRCNN(0)
    openpose = OpenPose(0)
    
    for datetime in datetimes:

        datetime_save_dir = dm.get_save_directory(datetime)

        if not dm.check_path_exists(datetime_save_dir):
            print('Making a save directory in: {}'.format(datetime_save_dir))
            os.makedirs(datetime_save_dir)
        else:
            continue

        rgb_path = dm.get_rgb_path(datetime)
        depth_path = dm.get_depth_path(datetime)

        # sort rgb files before looping
        # order matters!
        filenames = dm.get_sorted_rgb_images(datetime)
    
        # loop
        for fn in filenames:
            if fn.endswith(".png"): 
                print('\nimage: ', fn)

                # find the corresponding depth image
                rgb_img = os.path.join(rgb_path, fn)
                depth_img = os.path.join(depth_path, fn)
                if not os.path.exists(depth_img):
                    print('Could not find corresponding depth image in: {}'.format(depth_img))
                    continue

                out_q = mp.Queue()
                
                p_openpose = mp.Process(target=get_pose, args=(openpose, out_q, rgb_img, depth_img))
                p_maskrcnn = mp.Process(target=get_object, args=(maskrcnn, out_q, rgb_img, depth_img))

                p_maskrcnn.start()
                p_openpose.start()

                # wait till everything is over
                p_maskrcnn.join()
                p_openpose.join()

                poses. masks = [out_q.get() for i in range(2)].sort()

                print(poses)
                print(masks)

                filename = fn.split('.')[0] + '.npz'
                file_save_path = os.path.join(datetime_save_dir, filename)

                # save the files as npz
                if not poses:
                    if not masks:
                        continue
                else:
                    if not masks:
                        np.savez_compressed(file_save_path, poses=poses)
                    else:
                        np.savez_compressed(file_save_path, poses=poses, masks=masks)


                print("saved")
                



                





