import argparse
import os
import sys

import numpy as np

from utils import *

MAP_ROOT = os.path.join(ROOT, 'map_data')



def get_data_from_npz(file_path):
    '''Loads poses and objects dictionary from npz'''
    poses, objects = None, None
    data = np.load(file_path)
    files = data.files
    if len(files):
        if 'poses' in files:
            poses = data['poses'][()]  # because it's a dict
        
        if 'objects' in files:
            objects = data['objects'][()]
    
    return poses, objects


class MapManagement(FileManagement):
    
    def __init__(self, event=None, root_path=MAP_ROOT):
        '''
        Useful class for iterating through map data
        '''
        
        super().__init__()
        assert self.check_path_exists(root_path), "Path {} doesn't exists!".format(root_path)

        self.root = root_path

        if event is None:
            print('Should change event afterwards!')
        else:
            self.change_event(event)  # change event 

    def _get_event_path(self, loc, event):
        event_path = os.path.join(loc, event)
        return event_path
    
    def change_event(self, event):
        self.event = event

        # Set event path
        self.root_event_path = self._get_event_path(self.root, self.event)

        ## Get clips
        self.clips = self.natural_sort(os.listdir(self.root_event_path))

        print(f"Using Event {self.event}")
        print("Available clips")
        print(self.clips)

        return self.clips

    def get_clip_path(self, clip_name):
        ''''Get rgb path for specified clip'''
        return os.path.join(self.root_event_path, clip_name)

    def get_npz_files(self, clip_name):
        return os.listdir(self.get_clip_path(clip_name))

    def get_sorted_npz_files(self, clip_name):
        files = self.get_npz_files(clip_name)
        return self.natural_sort(files)
    
    def get_clip_directory(self, clip_name):
        return os.path.join(self.root_event_path, clip_name)
    
    def get_poses_objects_from_npz(self, file_path):
        return get_data_from_npz(file_path)


class MappingTool:
    
    def __init__(self):
        self.init_edges()
        self.sigma = 0.7
        self.gaussian_filter = self._init_gaussian_filter()
    
    def init_edges(self,
                   x_min=-2500,
                   y_min=-2500,
                   x_max=1500,
                   y_max=1500,
                   bin_x=100,
                   bin_y=100):
        
        # in milimeters
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.bin_x = bin_x  # 10cm
        self.bin_y = bin_y
        
        self.xedges = [i for i in range(self.x_min, self.x_max, self.bin_x)]
        self.yedges = [i for i in range(self.y_min, self.y_max, self.bin_y)]
         
    def normalize(self, H):
        '''Normalize matrix'''
        if H.max() != 0:
            H = H/H.max()
        return H
    
    def _init_gaussian_filter(self):
        '''Gaussian filter'''
        #FIXME: implement if scipy not available
        from scipy.ndimage.filters import gaussian_filter
        return gaussian_filter
        
    def single_histogram(self, points):
        '''
        Return histogram from points
        '''
        # get all grid points
        x = points[0][~np.isnan(points[0])]
        y = points[1][~np.isnan(points[1])]
        # z = points[2][~np.isnan(points[1])]

        # create histogram2d
        # returns H, xedges, yedges
        H, _, _ = np.histogram2d(x, y, bins=(self.xedges, self.yedges))
        
        # transpose
        H = H.T
        
        # gaussian filter
        H = self.gaussian_filter(H, sigma=self.sigma)

        # normalize
        H = self.normalize(H)

        return H


def process_body_part(pose, joints):
    '''
    return point for joints body part given joints
    joints = [array of joint values]
    '''
    # initialize
    points = np.empty((len(joints), 3))
    points[:] = np.nan
    
    for i, joint in enumerate(joints):
        point = pose[joint]
        
        if (point == [0,0,0]).all():
            continue
        
        points[i] = pose[joint]
        
    # change axis
    points = np.rollaxis(points, 1)
        
    return points 
    
    
def process_object(points):
    '''return points data ([x: [], y: [], z: []])'''
    points = np.rollaxis(points, 1)
    return points  


def save_maps_in_event(event, root_path, save_path):
    
    # file management
    manager = ClipsSavedDataManagement(event=event, root_path=root_path)
    # all of the clips
    clips = manager.change_event(event)
    
    # initialize mapping tool
    mappingtool = MappingTool()
    
    # create path
    event_path = os.path.join(save_path, f'{event}')
    if not manager.check_path_exists(event_path):
        os.mkdir(event_path)
        
    
    for clip in clips:
        
        # create path
        clip_path = os.path.join(event_path, clip)
        if not manager.check_path_exists(clip_path):
            os.mkdir(clip_path)
        
    
        all_files = manager.get_sorted_npz_files(clip)

        print(f'{clip}: ', len(all_files))

        for file in all_files:
            
            file_path = os.path.join(clip_path, file)
            
            if manager.check_path_exists(file_path):
                print(f'file {file} exists')
                continue
            
            # initialize averages
            pose_averages = {}
            object_averages = {}
            points = np.empty((3, 1))
            points[:] = np.nan
            H = mappingtool.single_histogram(points)
            for name, value in eventmap_pose_dict.items():
                pose_averages[name] = H.copy()
            for name, value in eventmap_object_dict.items():
                object_averages[name] = H.copy()

            filename = os.path.join(manager.get_clip_directory(clip), file)

            # get poses and masks
            poses, masks = poses_masks_from_npz(filename)

            # process poses
            if not(poses is None):
                for i, pose in poses.items():
                    # for each people
                    for name, joints in eventmap_pose_dict.items():
                        pose_points = process_body_part(pose, joints)
                        pose_H = mappingtool.single_histogram(pose_points)

                        pose_averages[name] += pose_H

            # process masks
            if not(masks is None):       
                for mask in masks:

                    object_id = int(mask.split('_')[0])

                    for object_name, object_ids in eventmap_object_dict.items():
                        if object_id in object_ids:
                            mask = masks[mask]

                            mask_points = process_object(mask)
                            mask_H = mappingtool.single_histogram(mask_points)

                            object_averages[object_name] += mask_H
        
            # save as npz
            np.savez_compressed(file_path, poses=pose_averages, objects=object_averages)


def save_for_all_events(map_path=None, ):
    
    SAVE_ROOT = os.path.join(save_clip_data, 'mask_pose')  # where pose and masks npz is located
    if map_path is None:
        map_path = os.path.join(ROOT, 'map_data')
    
    # choose an event
    events = [event_names[i] for i in event_ids]
    
    for event in events:
        save_maps_in_event(event, root_path=SAVE_ROOT, save_path=map_path)
            

def show_event_map():
    pass
            

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='maptool')
    parser.add_argument('--type', default='save', help="choose from 'save' or 'show'")
    args = parser.parse_args()
    
    if args.type == 'save':
        save_for_all_events()
        
    elif args.type == 'show':
        show_event_map()
        
    else:
        print('error')