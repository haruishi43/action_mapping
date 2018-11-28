import sys
import os

from utils import *

# Global Variables:
CLIPS_ROOT = raw_clip_data
MASK_ROOT = os.path.join(save_clip_data, 'mask_pose')


if __name__ == '__main__':
    
    
    events = [event_names[i] for i in event_ids]
    
    for event in events:
        dm = ShortClipManagement(event, CLIPS_ROOT, MASK_ROOT)
        