import os
import json
import time
from flask import Flask, render_template, Response, request
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

from camera import Camera


app = Flask(__name__)

def camera_gen(c):
    """Video streaming generator function."""
    while True:
        frame = c.get_frame()
        # save pose?
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(camera_gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pose_feed')
def pose_feed():
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

    # app.debug = True
    # server = pywsgi.WSGIServer(('localhost', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()