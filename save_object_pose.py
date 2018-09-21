import os
import argparse
import multiprocessing as mp
from datetime import datetime as dt
from collections import defaultdict

import numpy as np
import open3d as o3

from open3d_chain import Open3D_Chain
from getter_models import MaskRCNN, OpenPose, coco_label_names, JointType
from utils import DataManagement


def convert2world(coord, P):
    '''
    Convert from Camera coordniate to World coordinate (according to P)
    '''
    _coord = np.concatenate([np.asarray(coord), [1.000]])
    # For visualization, flip it in x-axis
    rotate = np.array([[1,0,0,0], [0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    n = rotate.dot(_coord)
    return P.dot(n)[:3]


def calc_xy(x, y, z, K):
    '''
    K: intrinsic matrix
    x: pixel value x
    y: pixel value y
    z: mm value of z
    '''

    fx = K[0][0]
    fy = K[1][1]
    u0 = K[0][2]
    v0 = K[1][2]

    _x = (x - u0) * z / fx
    _y = (y - v0) * z / fy
    return _x, _y


def downsample_nparray(arr, ratio=1):  # arr = [[x, y, z], [...] ]
    pcd = o3.PointCloud()
    pcd.points = o3.Vector3dVector(arr)
    downpcd = o3.voxel_down_sample(pcd, voxel_size = (10*ratio))  # 5 millimeters
    return np.asarray(downpcd.points)


def get_pose(depths, K, P, poses, scores):
    pose_dict = {}

    poses_num = 0
    for i, pose in enumerate(poses):

        if scores[i] < 4:  # scoring system is wierd (probably ~1 per joints?)
            continue
        
        joints = np.empty((len(JointType), 3))
        joints.fill(np.nan)

        for j, joint in enumerate(pose):
            x, y = int(joint[0]), int(joint[1])
            Z = depths[y][x]

            if Z != 0.0:
                # Depth = 0 means that depth data was unavailable
                X, Y = calc_xy(x, y, Z, K)
                joints[j] = convert2world(np.asarray([X, Y, Z]), P)
                # print('x: {}, y: {}, depth: {}'.format(X, Y, Z))

        pose_dict[poses_num] = joints
        poses_num += 1
    
    return pose_dict


def get_object(depths, K, P, labels, masks, scores):
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
            X, Y = calc_xy(x, y, Z, K)
            # print('x: {}, y: {}, depth: {}'.format(X, Y, Z))

            # append to points
            points[j] = convert2world(np.asarray([X, Y, Z]), P)
        
        downsampled_points = downsample_nparray(points, mask_size/image_size)
        
        title = str(label) + '_' + str(objects[label])
        object_masks[title] = downsampled_points
        objects[label] += 1

    return object_masks


if __name__ == "__main__":
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

    # camera params
    o3_chain = Open3D_Chain()
    K = o3_chain.get_K()
    P = o3_chain.get_P()

    # Intialize Models:
    maskrcnn = MaskRCNN(0)
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
                    print("File exists")
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
                
                # MASKRCNN
                _, labels, scores, masks = maskrcnn.predict(
                    rgb.swapaxes(2, 1).swapaxes(1, 0))

                dict_poses = get_pose(depths, K, P, poses, scores)
                dict_masks = get_object(depths, K, P, labels, masks, scores)

                
                # save the files as npz
                if not len(dict_poses):
                    if not len(dict_masks):
                        continue
                    else:
                        np.savez_compressed(file_save_path, masks=dict_masks)
                        print("saved")
                else:
                    if len(dict_masks):
                        np.savez_compressed(file_save_path, poses=dict_poses)
                    else:
                        np.savez_compressed(file_save_path, poses=dict_poses, masks=dict_masks)

                    print("saved")
                