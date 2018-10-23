import os
import sys
import time
import shutil
import numpy as np
from datetime import datetime as dt
from datetime import timedelta

''' 
# Objective:

Better management for my project\'s data format

# Directory format:
    root_dir:
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
rgb = 'rgb'
depth = 'depth'


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

    def get_rgb_path(self, datetime):
        ''''Get rgb path for specified date'''
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root, dir_name)
        return os.path.join(path, rgb)
    
    def get_depth_path(self, datetime):
        ''''Get depth path for specified date'''
        dir_name = self._datetime2string(datetime)
        path = os.path.join(self.root, dir_name)
        return os.path.join(path, depth)

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



def main():
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


if __name__ == '__main__':
    main()