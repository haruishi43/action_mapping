import os
import sys

import numpy as np
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import DataManagement, poses_masks_from_npz
from getter_models import *

sys.path.insert(0, 'openpose')  # append path of openpose
from entity import params, JointType

def main():
    dm = DataManagement()
    after = dt(2018, 9, 9, 13, 7, 0)
    before = dt(2018, 9, 9, 13, 16, 0)
    datetimes = dm.get_datetimes_in(after, before)

    # item(dm, datetimes)
    pose(dm, datetimes)


def pose(dm, datetimes):
    joint_val = JointType.RightHand.value
    print(JointType(joint_val))

    files = []
    for datetime in datetimes: 
        data_path = dm.get_save_directory(datetime)
        if os.path.exists(data_path):
            sorted_files = dm.natural_sort(os.listdir(data_path))  # sorted files
            files = files + [os.path.join(data_path, f) for f in sorted_files]

    print("number of file: ", len(files))
    joint = np.zeros((len(files), 3))
    for i, f in enumerate(files):
        poses, _ = poses_masks_from_npz(f)
        
        if not(poses is None):
            pose_ids = list(poses.keys()
            points = poses[pose_ids[0]]

            point = points[joint_val]
            if (point == [0,0,0]).all():
                joint[i] = np.asarray([np.nan, np.nan, np.nan])
            else:
                joint[i] = point

    joint.shape
    roll_joint = np.rollaxis(joint, 1)
    roll_joint.shape

    show_location_map(roll_joint, JointType(joint_val))


def item(dm, datetimes):
    ob_name = 'cup'
    
    files = []
    for datetime in datetimes:
        data_path = dm.get_save_directory(datetime)
        if os.path.exists(ob_path):
            ordered_files = dm.natural_sort(os.listdir(data_path))
            files = files + [os.path.join(ob_path, f) for f in ordered_files]
    
    print('number of files: ', len(files))

    mean_of_points = []
    for i, f in enumerate(files):
        _, masks = poses_masks_from_npz(f)

        if not(masks is None)
        item_index = coco_label_names.index(name)

        for mask in masks:
            
            # TODO:
            # get string name
            # check if the first digit is some class
            # if so, append the mean

            mean = points.mean(0)
            mean_of_points.append(mean)

    mp = np.asarray(mean_of_points)
    mp = np.rollaxis(mp, 1)

    # show_movement(mp)

    show_location_map(mp, ob_name)



def show_movement(points):

    plt.subplot(2, 2, 1)
    plt.plot(points[0], points[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.suptitle('Movement of object', fontsize=16)

    plt.subplot(2, 2, 2)
    plt.scatter(points[0], points[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def show_location_map(points, name):
    xedges = [i for i in range(-2000, 2000, 50)]
    yedges = [i for i in range(-2000, 2000, 50)]

    x = points[0][~np.isnan(points[0])]
    y = points[1][~np.isnan(points[1])]
    # z = points[2][~np.isnan(points[2])]

    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T


    fig = plt.figure()

    ax = fig.add_subplot(111, title=name,
                        aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    # non uniform image:

    # ax = fig.add_subplot(111, title='NonUniformImage: interpolated',
    #                      aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    # im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
    # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # im.set_data(xcenters, ycenters, H)
    # ax.images.append(im)
    plt.show()



if __name__ == '__main__':
    main()