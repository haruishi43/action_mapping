import sys
import os

import numpy as np
import cv2

from pyrs import PyRS
from utils import DataManagement



if __name__ == '__main__':

    w = 1280
    h = 720
    
    with PyRS(w=w, h=h) as pyrs:
        print('Modes:')
        print('\tExit:\tq')

        preset = pyrs.get_depths_preset()
        preset_name = pyrs.get_depths_preset_name(preset)
        print('Preset: ', pyrs.get_depths_preset_name(preset))

        events = ['eating', 'reading', 'using computer']
        count = 0

        event = events[0]
        root = "data"
        dm = DataManagement(root=root)
        #TODO: create directories

        event_path = os.path.join(root, event)
        rgb_path = os.path.join(event_path, 'rgb')
        depth_path = os.path.join(event_path, 'depth')
        if not dm.check_path_exists(event_path):
            print("creat directories")
            os.mkdir(event_path)
            os.mkdir(rgb_path)
            os.mkdir(depth_path)


        while True:
            # Wait for a coherent pair of frames: depth and color
            pyrs.update_frames()

            # Get images as numpy arrays
            color_image = pyrs.get_color_image()
            depths_image = pyrs.get_depths_frame()

            # Show image
            # cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RGB', color_image)

            # cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Depth', depths_image)


            key = cv2.waitKey(10)

            if key == ord('q'):
                # end OpenCV loop
                break

            # save rgb and depths
            rgb_file = rgb_path + "/" + str(count) + ".png"
            depth_file = depth_path + "/" + str(count) + ".png"
            cv2.imwrite(rgb_file, color_image) 
            cv2.imwrite(depth_file, depths_image)

            count += 1