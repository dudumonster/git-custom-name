# import pickle as pkl
# session = "/home/oseasy/GraduationProject/HEVI/val_one/201802061131000744_2.pkl"
# data = pkl.load(open(session, 'rb'))
# print("1")

# import numpy as np
# a = np.arange(20).reshape(4,5)
# a.tofile("a.bat",sep=",",format='%d')

import numpy as np
f = open("a.bat",'rb')
a = np.fromfile(f,dtype=np.int,sep=',',count=1)[0]
print(a)
w = int(np.fromfile(f, np.int32, sep=',', count=1))
h = int(np.fromfile(f, np.int32, sep=',', count=1))
print(w, h)
