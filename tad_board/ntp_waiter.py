import time
import socket
import struct

IP_ADDRESS = "192.168.62.117"
#IP_ADDRESS = '127.0.0.1'
PORT = 30301
#PORT = 8081

DATA_LEN = 8

now_time = time.time()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP_ADDRESS, PORT))

while True:
    try:
        data, addr = sock.recvfrom(DATA_LEN)
        time_2 = time.time()
    except Exception:
        print('Recv timeout')
        raise IOError
    else:
        print(addr)
        time_1 = struct.unpack('d',data[0:8])[0]
        time_3 = time.time()
        send_data = struct.pack('d',time_1) + struct.pack('d',time_2) + struct.pack('d',time_3)
        sock.sendto(send_data,("192.168.62.199",30300))
        #sock.sendto(send_data,("127.0.0.1",8081))
