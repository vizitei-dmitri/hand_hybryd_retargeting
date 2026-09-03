import dg5f_python
import numpy as np
import time
import serial
import threading

shared_value = [0,0,0,0,
                0,0,0,0,
                0,0,0,0,
                0,0,0,0,
                0,0,0,0]
lock = threading.Lock()
stop_event = threading.Event()

glove = serial.Serial("/dev/ttyUSB0", baudrate=115200)

hand = dg5f_python.DGApi.instance("169.254.186.72", 502, 1)

def reader():
    global shared_value, hand
    while not stop_event.is_set():
        with lock:
            ok, q = hand.get_current_current()
            if ok:
                shared_value = q

        time.sleep(0.0001)   # чтобы не сжигать CPU

t = threading.Thread(target=reader, daemon=True)

q_t1 = np.zeros(20, dtype=np.float32)
q_t2 = np.array([0,0,0,0,
                 0,0,50,50,
                 0,0,50,50,
                 0,0,50,50,
                 0,0,50,50], dtype=np.float32)

hand.start()
t.start()

hand.set_target_position(q_t1)

t = time.time()

while 1:
    a = glove.readline().decode("utf-8")
    try: 
        digits = list(map(float, a[1:-4].split(',')))
        digits[0] = 1*(digits[0]+30)
        digits[1] = -1*(digits[1] - 78)
        digits[2] = -1*(digits[2] - 17)
        digits[3] = -1*(digits[3] - 17)

        digits[5] *= -1
        digits[6] *= -1
        digits[7] *= -1

        digits[9] *= -1
        digits[10] *= -1
        digits[11] *= -1

        digits[12] *= 1
        digits[13] *= -1
        digits[14] *= -1
        digits[15] *= -1

        # digits[16] = -1*(digits[16]+15)
        digits[18] *= -1
        digits[19] *= -1

        digits = np.array(digits)

        with lock:
            hand.set_target_position(digits)
            print(shared_value)

    except:
        print("SOS")

# while 1:

#     if time.time() - t < 1:
#         hand.set_target_position(q_t1)
#     elif time.time() - t < 2:
#         hand.set_target_position(q_t2)
#     else:
#         t = time.time()
    
#     ok, q = hand.get_current_position()
#     if ok:
#         print(q)

hand.stop()