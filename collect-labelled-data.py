#!/usr/bin/env python3

import csv
import json
import socket
import sys
import time
import traceback

#################   Server Connection Code  ####################
     

fields = [
  "accelerometerAccelerationX",
  "accelerometerAccelerationY",
  "accelerometerAccelerationZ",
  "motionAttitudeReferenceFrame",
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

def main(label_name = "sitting", label_index = 0):
    try:
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

        print("opening csv file...")
        fout = open(label_name+"-data.csv", "w")
        fout_raw = open(label_name+"-raw.dat", "w")
        csvout = csv.writer(fout, delimiter=",")

        start_time = time.time()

        csvout.writerow(fields + ["activity"])
        count = 0
        while True:
            message = connection.recv(4096)
            rawdata = message.decode()
            fout_raw.write(rawdata)
            try:
                data = json.loads(rawdata)
            except json.decoder.JSONDecodeError:
                # got bad JSON from the watch
                sys.stderr.write("Received malformed JSON\n")
                traceback.print_exc()
                continue
            
            try:
                csvout.writerow([data[key] for key in fields] + [label_index])
            except KeyError:
                sys.stderr.write("Missing required field(s)\n")
                traceback.print_exc()
                continue

            count += 1

            if count % 10 == 0:
                time_since = int(time.time() - start_time)
                time_since = "%d:%02d" % (time_since // 60, time_since % 60)
                print(f"\rCollected {count} samples in {time_since}", end='') ; sys.stdout.flush()
    except KeyboardInterrupt: 
        # occurs when the user presses Ctrl-C
        print("\nUser Interrupt. Quitting...")
    finally:
        print("\nclosing csv file")
        fout.close()
        fout_raw.close()

        print("closing socket for receiving data")
        try:
            receive_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            # will get raised if the socket was already shutdown
            pass
        receive_socket.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage:\n\t%s <activity label> <activity index>" % sys.argv[0])
    else:
        main(sys.argv[1], int(sys.argv[2]))
