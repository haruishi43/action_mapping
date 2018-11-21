from utils import *

CLIPS_ROOT = raw_clip_data
SAVE_ROOT = os.path.join(save_clip_data, 'mask_pose')




if __name__ == '__main__':
    
    event = events[2]  # coffee
    clip_id = 0
    
    events = [event_names[i] for i in event_ids]
    
    
    # file manager 
    manager = ClipsSavedDataManagement(event=coffee, root_path=SAVE_ROOT)
    clips = manager.change_event(coffee)
    
    clip = clips[clip_id]  # choose first clip