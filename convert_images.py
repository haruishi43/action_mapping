import os
import sys

import numpy as np
from PIL import Image

from utils import *


def resize_image(image, size=(240, 135), interp=Image.BILINEAR):
    '''
    input: PIL image
    output: resized PIL image
    '''
    return image.resize(size=size, resample=interp)



def for_each_event(event, clip_root, save_root):
    
    dm = ShortClipManagement(root=clip_root, save_path=save_root)
    clips = dm.change_event(event)
    
    save_event_path = dm.get_save_event_directory()
    
    for clip in clips:
        rgb_images = dm.get_sorted_rgb_images(clip)
        
        save_clip_path = dm.get_save_clip_directory(clip)
        
        print(save_clip_path)

        for image_name in rgb_images:
            image_path = os.path.join(dm.get_rgb_path(clip), image_name) 
            save_image_name = image_name.split('.')[0] + '.jpg'
            save_image_path = os.path.join(save_clip_path, save_image_name)
            
            if dm.check_path_exists(save_image_path):
                print(f'image {image_name} exists')
                continue
            
            image = Image.open(image_path)
            resized_image = resize_image(image)
            
            resized_image.save(save_image_path)


if __name__ == '__main__':
    
    clip_root = raw_clip_data
    save_root = os.path.join(ROOT, 'jpg')
    
    events = [event_names[i] for i in event_ids]
    
    for event in events:
        
        for_each_event(event, clip_root, save_root)
        