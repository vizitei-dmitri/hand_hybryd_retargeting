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

hand.start()
t.start()

hand.set_target_position(dg5f_python.poses.grasps.OPEN_BIG)

tm = time.time()

while 1:
    try:
        if time.time() - tm < 1:
            with lock: 
                hand.set_target_position(dg5f_python.poses.grasps.OPEN_BIG)
        elif time.time() - tm < 2:
            with lock: 
                hand.set_target_position(dg5f_python.poses.numbers.ONE)
        else:
            tm = time.time()
        
        print(shared_value)

    except KeyboardInterrupt:
        stop_event.set()
        hand.stop()
        t.join()
        break