#!/bin/sh

# python rendering.py

# find window ID
WINID=$(xdotool search --onlyvisible --name pose)

# move the window
xdotool windowmove $WINID 2000 10


# find window size
xdotool windowsize $WINID 1920 1080

