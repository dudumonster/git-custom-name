import time
import socket
import struct
import threading

IP_ADDRESS = "192.168.62.117"
PORT = 30301

DATA_LEN = 24

delay = 0

expTime = 0

def recv_thread(sock):
    index = 0
    while(index < 100):
        try:
            data, addr = sock.recvfrom(DATA_LEN)
            time_4 = time.time()
        except Exception:
            print('Recv timeout')
            raise IOError
        else:
            time_1 = struct.unpack('d',data[0:8])[0]
            time_2 = struct.unpack('d',data[8:16])[0]
            time_3 = struct.unpack('d',data[16:24])[0]

            print("Recv")

            global delay
            global expTime
            delay +=((time_4 - time_1) - (time_3 - time_2))/2
            expTime += ((time_4 - time_3) - (time_2 - time_1))/2
            index += 1


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP_ADDRESS, PORT))

t1 = threading.Thread(target=recv_thread, args=(sock,))
t1.start()

for i in range(100):
    time.sleep(0.02)
    now_time = time.time()
    send_data = struct.pack('d',now_time)
    sock.sendto(send_data,("192.168.62.199",30300))

    print("sending!")

t1.join()

print("Delay: ", delay/100)
print("ExpTime: ",expTime/100)
