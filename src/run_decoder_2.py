# server code
import time

import pyautogui
import zmq
import decoder_2
import struct
import numpy as np
import pandas as pd
from config import load_parameters
from SimpleESN import SimpleESN

# parameters
from keypress import AutoKeyPressor
from decoder_2 import Inference

HOST = '127.0.0.1'
PORT_APP = 9999
PORT_EEG = 9998
eeg_channels = [1, 2, 3, 4, 5, 6]
load_parameters(globals())

# parameters, not loaded by config.json
ch_num = len(eeg_channels)
data_shape = (ch_num, -1)


# socket 수신
def decoder_thread():

    # open sockets
    context = zmq.Context()
    app_socket = context.socket(zmq.PUSH)
    app_socket.bind(f'tcp://{HOST}:{PORT_APP}')  # socket bind to pong app (push)
    eeg_socket = context.socket(zmq.PULL)
    eeg_socket.bind(f'tcp://{HOST}:{PORT_EEG}')  # socket bind to eeg measuring app (pull)

    while True:  # break app with keyboard interrupt (ctrl+c)

        # pulling bytes from eeg measuring app
        try:
            # data = eeg_socket.recv(flags=zmq.NOBLOCK) # non-blocking
            data = eeg_socket.recv()  # blocked until signal comes

        # except zmq.ZMQError: continue   # no socket connection from eeg-app (used for non-blocking)
        except Exception as e:
            raise e  # errors

        ###########################################
        # make np array from bytes
        data = np.frombuffer(data, dtype=float).copy().reshape(data_shape)

        # print for debug
        print('data.shape = ', data.shape)  # (16,62)
        data = pd.DataFrame(data)
        data = data.iloc[:6,:] # using 6chan
        print(data)
        # print('data[0,0] = ', data[0, 0])
        # print('data.sum(axis=1) = ', data.sum(axis=1))
        print('####################')

        # decoding
        data = decoder_2.filtering_bandpass_update(data)
        print('data.shape =', data.shape)

        output = Inference(data)
        print('data.shape =', output.shape)
        print(output) # model inference 잘 되는지 디버깅해보기.
        output = sum(output) / len(output)
        print(output)
        # assert output.shape == (1,)  # make sure the model output is one value
        # data = float(data[0])
        print('model output : ', output)


        # non-blocking pushing the model output to pong app
        try:
            app_socket.send(struct.pack('f', output), flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            continue  # if no socket connection (from pong-app)
        except Exception as e:
            raise e  # errors


if __name__ == '__main__':
    decoder_thread()

# ex_list = []
# timecounter = 0
# select_win = pyautogui.getWindowsWithTitle('Mind Pong')[0]  # run app으로 실행된 최소화된 창 선택
# if select_win.isMinimized:  # 창이 최소화되어있다면
#     select_win.restore()  # 창 복원
# select_win.activate()  # 창 활성화
# print(select_win.left, select_win.right, select_win.top, select_win.bottom)
# print(pyautogui.position())
# pyautogui.click(1288, 920)  # game 버튼 누름 # 1311,909
# print(pyautogui.position())
# pyautogui.click(1288, 920)  # game 버튼 누름
#
# autokp = AutoKeyPressor(ex_list)
#
# while True:
#     if timecounter <= 1:
#
#         timecounter += 1
#         time.sleep(1)
#     else:
#         timecounter = 0
#         c = np.random.randint(-1, 2)
#         print(c)
#         autokp.appendData(c)
#         print(ex_list)
#
#     autokp.update()
