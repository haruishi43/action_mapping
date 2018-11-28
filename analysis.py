import os
import sys

import numpy as np
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *
from getter_models import *


def main():
    dm = DataManagement()
    after = dt(2018, 9, 13, 19, 8, 0)
    before = dt(2018, 9, 13, 19, 9, 0)
    datetimes = dm.get_datetimes_in(after, before)

    # item(dm, datetimes)
    pose(dm, datetimes)


def pose(dm, datetimes):
    joint_val = JointType.Nose.value
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
            pose_ids = list(poses.keys())
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
    ob_name = 'potted plant'
    
    files = []
    for datetime in datetimes:
        data_path = dm.get_save_directory(datetime)
        if os.path.exists(data_path):
            ordered_files = dm.natural_sort(os.listdir(data_path))
            files = files + [os.path.join(data_path, f) for f in ordered_files]
    
    print('number of files: ', len(files))

    mean_of_points = []
    for i, f in enumerate(files):
        _, masks = poses_masks_from_npz(f)

        if masks is None:
            continue

        item_index = coco_label_names.index(ob_name)

        for mask in masks:
            
            # TODO:
            # get string name
            # check if the first digit is some class
            # if so, append the mean
            id = int(mask.split('_')[0])

            if id == item_index:
                mean = masks[mask].mean(0)
                mean_of_points.append(mean)

    mp = np.asarray(mean_of_points)

    if mp.any() == False:
        print("no such object ", ob_name, " detected")
        return

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

    
# in milimeters
y_min = -2500
y_max = 1500
x_min = -2500
x_max = 1500
bin_x = 100
bin_y = 100

sigma = 0.7

xedges = [i for i in range(x_min, x_max, bin_x)]
yedges = [i for i in range(y_min, y_max, bin_y)]
    
def show_location_map_new(points, name):
    xedges = [i for i in range(x_min, x_max, bin_x)]
    yedges = [i for i in range(y_min, y_max, bin_y)]

    x = points[0][~np.isnan(points[0])]
    y = points[1][~np.isnan(points[1])]
    z = points[2][~np.isnan(points[2])]
    
    # empty.
#     x = []
#     y = []
    
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    
    # gaussian filter
    H = gaussian_filter(H, sigma=sigma)
    print(H.max())
    
    # normalize
    if H.max() != 0:
        H = H/H.max()
    
    # color map?
    cmap = mpl.colors.ListedColormap(['grey','red'])

    fig = plt.figure()

    ax = fig.add_subplot(111, title=name,
                        aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H, cmap=colormap, vmin=0.0, vmax=1.0)
    # ax.pcolor(X, Y, H)

    plt.show()


def calc_mean_object():
    mean_of_points = []
    for i, f in enumerate(files):
        filename = os.path.join(manager.get_clip_directory(clip), f)
        _, masks = poses_masks_from_npz(filename)

        if masks is None:
            continue

        item_number = [i for i, j in object_dict.items() if j == ob_name]

        for mask in masks:

            # TODO:
            # get string name
            # check if the first digit is some class
            # if so, append the mean
            id = int(mask.split('_')[0])
            if id in item_number:
                mean = masks[mask].mean(0)
                mean_of_points.append(mean)

    mp = np.asarray(mean_of_points)
    if mp.any() == False:
        print("no such object ", ob_name, " detected")
        mp = np.empty((3, 1))
        mp[:] = np.nan
    else:
        mp = np.rollaxis(mp, 1)

        
def save_location_map(points, name):
    xedges = [i for i in range(x_min, x_max, bin_x)]
    yedges = [i for i in range(y_min, y_max, bin_y)]

    x = points[0][~np.isnan(points[0])]
    y = points[1][~np.isnan(points[1])]
    # z = points[2][~np.isnan(points[2])]

    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    
    H = gaussian_filter(H, sigma=sigma)  # gaussian blur
    H = H/H.max()  # normalize from 0 - 1 (1 is H.max())
    
    H_pix = (H * 255.9).astype(np.uint8)
    
    # y-axis becomes lowest to highest
    H_flipped = np.flipud(H_pix)
    
    img = Image.fromarray(H_flipped)
    img.save(f'{name}.png')
    
    
if __name__ == '__main__':
    main()