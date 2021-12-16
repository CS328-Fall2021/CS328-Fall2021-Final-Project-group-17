# -*- coding: utf-8 -*-
"""
This Python script receives incoming unlabelled accelerometer data through 
the server and uses your trained classifier to predict its class label.

"""

import socket
import sys
import json
import threading
import traceback
import numpy as np
import pickle
from features import extract_features # make sure features.py is in the same directory
from util import reorient, reset_vars

class_names = ["not touching face", "TOUCHING FACE"]

fields = [
  "accelerometerAccelerationX",
  "accelerometerAccelerationY",
  "accelerometerAccelerationZ",
#  "motionAttitudeReferenceFrame",  # this just gave us nan values
  "motionGravityX",
  "motionGravityY",
  "motionGravityZ",
  "motionHeading",
  "motionMagneticFieldAccuracy",
  "motionMagneticFieldX",
  "motionMagneticFieldY",
  "motionMagneticFieldZ",
  "motionPitch",
  "motionQuaternionW",
  "motionQuaternionX",
  "motionQuaternionY",
  "motionQuaternionZ",
  "motionRoll",
  "motionRotationRateX",
  "motionRotationRateY",
  "motionRotationRateZ",
  "motionUserAccelerationX",
  "motionUserAccelerationY",
  "motionUserAccelerationZ",
  "motionYaw",
]

count = 0

# Loading the classifier that you saved to disk previously
with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)
    
if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
def onActivityDetected(activity):
    """
    Notifies the user of the current activity
    """
    print("Detected activity:" + activity)

def predict(window):
    """
    Given a window of accelerometer data, predict the activity label. 
    """

    _, features = extract_features(window)
    activity_idx = int(classifier.predict(features.reshape(1, -1))[0])
    onActivityDetected(class_names[activity_idx])
    
    return
    

#################   Server Connection Code  ####################

#This socket is used to receive data from the data collection server
receive_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# binding to 0.0.0.0 instead of socket.gethostname(), which will
# only accept connections from localhost
print("Waiting for connection...", end='') ; sys.stdout.flush()
receive_socket.bind(("0.0.0.0", 9800))
receive_socket.listen(4) # become a server socket, maximum 5 connections
connection, address = receive_socket.accept()
print('connected')

try:
    sensor_data = []
    window_size = 4 # ~1 sec assuming 25 Hz sampling rate
    step_size = 4 # no overlap
    index = 0 # to keep track of how many samples we have buffered so far
    reset_vars() # resets orientation variables
        
    while True:
        try:
            message = connection.recv(4096)
            rawdata = message.decode()
            try:
                data = json.loads(rawdata)
            except json.decoder.JSONDecodeError:
                sys.stderr.write("Received malformed JSON\n")
                traceback.print_exc()
                continue

            row = [float(data[key]) for key in fields]

            sensor_data.append(row)
            index+=1
            # make sure we have exactly window_size data points :
            while len(sensor_data) > window_size:
                sensor_data.pop(0)
                
            if (index >= step_size and len(sensor_data) == window_size):
                t = threading.Thread(target=predict, args=(np.asarray(sensor_data[:]),))
                t.start()
                index = 0
                
            sys.stdout.flush()
        except KeyboardInterrupt: 
            # occurs when the user presses Ctrl-C
            print("User Interrupt. Quitting...")
            break
        except Exception as e:
            # ignore exceptions, such as parsing the json
            # if a connection timeout occurs, also ignore and try again. Use Ctrl-C to stop
            # but make sure the error is displayed so we know what's going on
            if (str(e) != "timed out"):  # ignore timeout exceptions completely       
                print(e)
            pass
except KeyboardInterrupt: 
    # occurs when the user presses Ctrl-C
    print("User Interrupt. Qutting...")
finally:
    print('closing socket for receiving data')
    try:
        receive_socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        # will get raised if the socket was already shutdown
        pass
    receive_socket.close()
