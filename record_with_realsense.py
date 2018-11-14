import sys
import os
import argparse

import numpy as np
import cv2

from pyrs import PyRS
from utils import DataSaver, ShortClipSaver, event_names, event_ids



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Pose Getter')
    parser.add_argument('--event', type=int, default=1, help='Event ID')
    args = parser.parse_args()
    event_id = args.event
    assert event_id in event_ids, "Event should not be saved!"
    event_name = event_names[event_id]

    w = 1280
    h = 720

    print(f"saving for {event_name}")
    root = "data"
    dm = DataSaver(event=event_name)  # segmented by time
    # dm = ShortClipSaver(event=event_name)  # for saving short clips
    
    with PyRS(w=w, h=h) as pyrs:
        print('Modes:')
        print('\tExit:\tq')

        preset = pyrs.get_depths_preset()
        preset_name = pyrs.get_depths_preset_name(preset)
        print('Preset: ', pyrs.get_depths_preset_name(preset))

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

            key = cv2.waitKey(1)

            if key == ord('q'):
                # end OpenCV loop
                break

            # save rgb and depths
            rgb_file, depth_file = dm.get_rgb_depth_filename()
            cv2.imwrite(rgb_file, color_image) 
            cv2.imwrite(depth_file, depths_image)
