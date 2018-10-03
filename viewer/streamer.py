import os
import sys
import cv2
import threading
import socket
import struct
import io
import json
import numpy as np



class Streamer(threading.Thread):


    def __init__(self, hostname, port):
        threading.Thread.__init__(self)

        self.hostname = hostname
        self.port = port
        self.connected = False
        self.jpeg = None


    def run(self):

        self.isRunning = True

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        s.bind((self.hostname, self.port))
        print('Socket bind complete')

        data = ""
        payload_size = struct.calcsize("L")

        s.listen(10)
        print('Socket now listening')

        while self.isRunning:

            conn, addr = s.accept()

            while True:

                data = conn.recv(4096).decode('utf-8')

                if data:
                    print(data)
                    frame = np.asarray(json.loads(data))

                    ret, jpeg = cv2.imencode('.jpg', frame)
                    self.jpeg = jpeg

                    self.connected = True

                else:
                    conn.close()
                    self.connected = False
                    break

        self.connected = False

    def stop(self):
        self.isRunning = False

    def client_connected(self):
        return self.connected

    def get_jpeg(self):
        return self.jpeg.tobytes()