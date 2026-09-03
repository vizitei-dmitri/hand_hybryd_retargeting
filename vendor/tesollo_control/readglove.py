import serial
import socket

glove = serial.Serial("/dev/ttyUSB0", baudrate=115200)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def array2str(arr):
    msg = "["
    for i in arr:
        msg = msg + str(i) + ","
    msg = msg[:-1] + "]"
    return msg

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

        digits[12] *= -1
        digits[13] *= -1
        digits[14] *= -1
        digits[15] *= -1

        # digits[16] = -1*(digits[16]+15)
        digits[18] *= -1
        digits[19] *= -1

        print(digits)
        sock.sendto(array2str(digits).encode('utf-8'), ("127.0.0.1", 8081))

    except:
        print("SOS")
    