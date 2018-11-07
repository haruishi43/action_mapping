import os
import sys
import time
import shutil
import numpy as np
from datetime import datetime as dt
from datetime import timedelta

#######################################################################################################
# Parameters
#######################################################################################################

event_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
event_names = ['other', 'eating', 'meeting', 'coffee break', 'sleeping', 'cooking', 'working', 'party', 'tending to plants', 'test']

object_dict = {
    1: 'person',
    27: 'bag', 31: 'bag', 33: 'bag',
    44: 'drink', 46: 'drink', 47: 'drink',
    48: 'utensil', 49: 'utensil', 50: 'utensil',
    51: 'bowl',
    52: 'food', 53: 'food', 54: 'food', 55: 'food', 59: 'food', 60: 'food', 61: 'food',
    62: 'chair',
    64: 'potted plant',
    67: 'dining table',
    73: 'laptop',
    77: 'cell phone',
    78: 'microwave',
    81: 'sink',
    82: 'refridgerator',
    84: 'book'
}

extracting_ids = [1, # person 
    27, # backpack
    31, 33, # handbag, suitcase
    44, # bottle\
    46, 47, 48, 49, 50, # wine glass, cup, fork, knife, spoon
    51, 52, 53, 54, 55, # bowl, banana, apple, sandwich, orange
    59, 60, # pizza, donut
    61, 62, 64, # cake, chair, potted plant
    67, # dining table
    73, # laptop
    77, 78, # cell phone, microwave
    81, 82, 84] # sink, refrigerator, book

# 80
# labels that can be obtained
coco_label_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, \
    57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


coco_label_names = ('background',  # class zero
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', #5
    'bus', 'train', 'truck', 'boat', 'traffic light', #10
    'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', #15 
    'bird', 'cat', 'dog', 'horse', 'sheep', #20
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', #25 
    'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', #30
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', #35
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', #40
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', #45
    'wine glass', 'cup', 'fork', 'knife', 'spoon', #50
    'bowl', 'banana', 'apple', 'sandwich', 'orange', #55
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', #60
    'cake', 'chair', 'couch', 'potted plant', 'bed', #65
    'mirror', 'dining table', 'window', 'desk','toilet', #70
    'door', 'tv', 'laptop', 'mouse', 'remote', #75
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', #80
    'sink', 'refrigerator', 'blender', 'book', 'clock', #85
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' #90
)

coco_label_colors = {
    1: [1, 0.7, 0],
    2: [0, 0, 0],
    3: [0, 0, 0],
    4: [0, 0, 0],
    5: [0, 0, 0],
    6: [0, 0, 0],
    7: [0, 0, 0],
    8: [0, 0, 0],
    9: [0, 0, 0],
    10: [0, 0, 0],
    11: [0, 0, 0],
    13: [0, 0, 0],
    14: [0, 0, 0],
    15: [0, 0, 0],
    16: [0, 0, 0],
    17: [0, 0, 0],
    18: [0, 0, 0],
    19: [0, 0, 0],
    20: [0, 0, 0],
    21: [0, 0, 0],
    22: [0, 0, 0],
    23: [0, 0, 0],
    24: [0, 0, 0],
    25: [0, 0, 0],
    27: [0.5, 0, 0.5],
    28: [0.3, 0.6, 1],
    31: [0.8, 0, 0.1],
    32: [0, 0.9, 1],
    33: [0.2, 0.2, 1],
    34: [0, 0, 0],
    35: [0, 0, 0],
    36: [0.1, 0.4, 0],
    37: [0, 0, 0],
    38: [0, 0, 0],
    39: [0, 0, 0],
    40: [0, 0, 0],
    41: [0, 0, 0],
    42: [0, 0, 0],
    43: [0, 0, 0],
    44: [0.9, 0.7, 1],
    46: [0, 0.2, 0.6],
    47: [1, 0.4, 0.5],
    48: [0, 0.1, 0.5],
    49: [0.2, 1, 0.2],
    50: [0.4, 0.7, 0.7],
    51: [0, 0, 0.3],
    52: [0, 0.5, 0.1],
    53: [0.1, 0.7, 0.3],
    54: [0.6, 0.5, 0.4],
    53: [0.3, 0.2, 0.1],
    54: [0.1, 0.2, 0.3],
    55: [0.4, 0.5, 0.6],
    56: [0.9, 0.8, 0.7],
    57: [0.6, 0.7, 1],
    58: [0.1, 0.1, 0.3],
    59: [0, 1, 0.5],
    60: [0.5, 0.3, 0.8],
    61: [0.6, 0.3, 0.1],
    62: [0.1, 0.6, 0.8],
    63: [1, 0.2, 0.6],
    64: [1, 0, 0.6],
    65: [0.9, 0.1, 0.9],
    67: [0.8, 0.3, 0.8],
    70: [0.4, 0.3, 0.9],
    72: [0, 0.3, 0.3],
    73: [0, 0.7, 1],
    74: [0, 0.5, 0.5],
    75: [1, 0.3, 0.2],
    76: [0.4, 0.4, 1],
    77: [0.1, 0.4, 0.3],
    78: [0, 0, 0.5],
    79: [0.7, 0.7, 0.3],
    80: [0.4, 0.3, 0],
    81: [0.8, 0.5, 0.3],
    82: [0.6, 0.9, 0.3],
    83: [0.5, 0.6, 0.2],
    84: [0.3, 0.6, 0.6],
    85: [0.9, 0.5, 0.1],
    86: [0.3, 0.5, 0.5],
    87: [1, 1, 0.3],
    88: [0.8, 1, 1],
    89: [0.3, 0.5, 0.3],
    90: [0.7, 0.8, 0.2]
}


#######################################################################################################
# File managements
#######################################################################################################
''' 
# Objective:

Better management for my project\'s data format

# Directory format:
    root_dir:
        -Event:
            - DATE_TIME:
                - rgb:
                    - files
                - depth:
                    -files

## Details:

Directory name:

- DATE_TIME: "%Y%m%d_%H%M"
    - separated by minutes

# TODO:

- Getting data and saving data differs...
- Data path is always not in this format

'''

raw_data = '/mnt/extHDD/raw_data/'
save_data = '/mnt/extHDD/save_data/'
RGB = 'rgb'
DEPTH = 'depth'


class FileManagement:
    def __init__(self):
        pass

    def check_path_exists(self, path):
        return os.path.exists(path)

    def get_subdirs(self, path):
        if self.check_path_exists(path):
            return os.listdir(path)
        else:
            print("Path {} doesn't exists!".format(path))
            return None

    def natural_sort(self, files):
        '''Natural sort array: note that each file has to have an integer'''
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        return files

    def _string2datetime(self, name):
        '''converts string to datetime format'''
        assert len(name) == 13, 'directory name {} is not 13 characters'.format(name)
        #FIXME: an better way?
        d, t    = name.split('_')
        _d      = [d[:4], d[4:6], d[6:]]
        _t      = [t[:2], t[2:]]
        Y, M, D = [int(a) for a in _d]
        h, m    = [int(a) for a in _t]
        # print(Y, M, D, "_", h, m)
        return dt(year=Y,month=M,day=D,hour=h,minute=m)

    def _datetime2string(self, datetime):
        '''converts datetime format to string'''
        assert type(datetime) is dt, "Input should be in {} format".format(dt)
        return datetime.strftime("%Y%m%d_%H%M")

    def remove_directories(self, path):
        shutil.rmtree(path)


class DataSaver(FileManagement):
    '''Save rgb and depth image'''
    def __init__(self, event, save_path=raw_data):
        super().__init__()
        self.data_root = save_path
        self.event_path = os.path.join(save_path, event)
        self._create_event_dir(event)

        self.current_dt = dt.now()
        self._create_datetime_dir()
        self.count = 0

    def _check_time(self):
        if dt.now().minute - self.current_dt.minute > 0:
            self._update_datetime()
            self._create_datetime_dir()

    def _update_datetime(self):
        self.count = 0
        self.current_dt = dt.now()

    def _create_datetime_dir(self):
        datetime = self._datetime2string(self.current_dt)
        datetime_path = os.path.join(self.event_path, datetime)
        if not self.check_path_exists(datetime_path):
            os.mkdir(datetime_path)
            self._also_create_image_path(datetime_path)
            print(f"Created directory in {datetime_path}")

    def _also_create_image_path(self, datetime_path):
        rgb_path = os.path.join(datetime_path, RGB)
        depth_path = os.path.join(datetime_path, DEPTH)
        os.mkdir(rgb_path)
        os.mkdir(depth_path)

    def get_rgb_depth_filename(self):
        self._check_time()
        datetime = self._datetime2string(self.current_dt)
        full_path = os.path.join(self.event_path, datetime)

        rgb_path = os.path.join(full_path, RGB)
        depth_path = os.path.join(full_path, DEPTH)
        rgb_name = os.path.join(rgb_path, str(self.count)+'.png')
        depth_name = os.path.join(depth_path, str(self.count)+'.png')
        self.count += 1

        return rgb_name, depth_name
        
    def _create_event_dir(self, event):
        event_path = os.path.join(self.data_root, event)
        if not self.check_path_exists(event_path):
            os.mkdir(event_path)
            print(f"Created directory in {event_path}")


class DataManagement(FileManagement):
    def __init__(self, root=raw_data, save_root=save_data):
        super().__init__()
        assert self.check_path_exists(root), "Path {} doesn't exists!".format(root)
        assert self.check_path_exists(save_root), "Path {} doesn't exists!".format(save_root)
        self.root = root
        self.save_root = save_root
        self.datetime_dirs = self.natural_sort(os.listdir(self.root))
        self.datetimes = [self._string2datetime(n) for n in self.datetime_dirs]

    def get_save_directory(self, datetime):
        return os.path.join(self.save_root, self._datetime2string(datetime))

    def get_datetime_dirs(self):
        '''returns a list of strings...'''
        return self.datetime_dirs
    
    def get_datetimes(self):
        '''returns a sorted list of datetime'''
        return sorted(self.datetimes)

    def get_datetime_subdirs(self, datetime, base_path=raw_data):  #FIXME:
        dir_name = self._datetime2string(datetime)
        path = os.path.join(base_path, dir_name)
        return self.get_subdirs(path)

    # dep
    def get_rgb_path(self, datetime):
        ''''Get rgb path for specified date'''
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root, dir_name)
        return os.path.join(path, RGB)
    
    def get_depth_path(self, datetime):
        ''''Get depth path for specified date'''
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root, dir_name)
        return os.path.join(path, DEPTH)

    def get_rgb_images(self, datetime):
        return os.listdir(self.get_rgb_path(datetime))

    def get_sorted_rgb_images(self, datetime):
        files = self.get_rgb_images(datetime)
        return self.natural_sort(files)

    def get_datetimes_in(self, after=dt.min, before=dt.now()):
        '''Get datetimes between two datetimes'''
        assert before - after > timedelta(seconds=0)
        datetimes = np.asarray(self.get_datetimes())
        return [t for t in datetimes if t >= after and t < before]


class EventDataManagement(FileManagement):

    def __init__(self, event, root=raw_data, save_path=save_data):
        super().__init__()
        assert self.check_path_exists(root), "Path {} doesn't exists!".format(root)
        assert self.check_path_exists(save_path), "Path {} doesn't exists!".format(save_path)

        self.root = root
        self.save_path = save_path

        self.change_event(event)  # change event 

    def _get_event_path(self, loc, event):
        event_path = os.path.join(loc, event)
        return event_path
    
    def change_event(self, event):
        self.event = event

        # Set event path
        self.root_event_path = self._get_event_path(self.root, self.event)
        self.save_event_path = self._get_event_path(self.save_path, self.event)

        # Datetimes for event
        self.datetime_dirs = self.natural_sort(os.listdir(self.root_event_path))
        self.datetimes = [self._string2datetime(n) for n in self.datetime_dirs]

        print(f"Using Event {self.event}")
        print("Available datetimes")
        print(sorted(self.datetimes))

    def get_rgb_path(self, datetime):
        ''''Get rgb path for specified date'''
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root_event_path, dir_name)
        return os.path.join(path, RGB)

    def get_depth_path(self, datetime):
        ''''Get depth path for specified date'''
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root_event_path, dir_name)
        return os.path.join(path, DEPTH)

    def get_rgb_images(self, datetime):
        return os.listdir(self.get_rgb_path(datetime))

    def get_sorted_rgb_images(self, datetime):
        files = self.get_rgb_images(datetime)
        return self.natural_sort(files)

    def get_datetimes(self):
        '''returns a sorted list of datetime'''
        return sorted(self.datetimes)

    def get_datetimes_in(self, after=dt.min, before=dt.now()):
        '''Get datetimes between two datetimes'''
        assert before - after > timedelta(seconds=0)
        datetimes = np.asarray(self.get_datetimes())
        return [t for t in datetimes if t >= after and t < before]

    def get_save_directory(self, datetime):
        return os.path.join(self.save_event_path, self._datetime2string(datetime))



#######################################################################################################
# Supporting Functions
#######################################################################################################

def poses_masks_from_npz(file_path):
    '''Loads pose and mask dictionary from npz'''
    masks, poses = None, None
    data = np.load(file_path)
    files = data.files
    if len(files):
        if 'poses' in files:
            poses = data['poses'][()]  # because it's a dict
        
        if 'masks' in files:
            masks = data['masks'][()]
    
    return poses, masks


def poses_objects_from_npz(file_path):
    '''Loads pose and mask dictionary from npz'''
    poses, bbox, center = None, None, None
    data = np.load(file_path)
    files = data.files
    if len(files):
        if 'poses' in files:
            poses = data['poses'][()]  # because it's a dict
        
        if 'bbox' in files:
            bbox = data['bbox'][()]

        if 'center' in files:
            center = data['center'][()]
    
    return poses, bbox, center


#######################################################################################################
# Tests
#######################################################################################################

def test():
    # demo:

    dm = DataManagement()
    datetimes = dm.get_datetimes()

    # for dt in datetimes:
    #     print(dt)
    #     print("\t", len(dm.get_rgb_images(dt)))

    after = dt(2018, 7, 23, 14, 0, 0)
    before = dt(2018, 7, 23, 15, 0, 0)
    between = dm.get_datetimes_in(after, before)

    print(len(between))

def test_events():

    dm = EventDataManagement(event='test', root=raw_data, save_path='data')

    datetimes = dm.get_datetimes()
    after = dt(2018, 11, 7, 16, 50, 0)
    before = dt(2018, 11, 7, 17, 0, 0)
    between = dm.get_datetimes_in(after, before)

    print(len(between))

    print(dm.get_sorted_rgb_images(between[0]))


if __name__ == '__main__':
    test_events()