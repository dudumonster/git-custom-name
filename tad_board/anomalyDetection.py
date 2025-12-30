import tkinter as tk

import OBU_Recv_Video
import RSU_Send_Video

IP = "10.247.51.2"
# IP = "127.0.0.1"
port = 30300
# port = 8081

def login(master):
    willSend = False
    loginFrame = tk.Frame(master)
    loginFrame.grid(padx=15,pady=15)
 
    ipLabel = tk.Label(loginFrame,text='IP').grid(column=1,row=1,columnspan=2)
    ipEntry = tk.Entry(loginFrame,)
    ipEntry.grid(column=3,row=1,columnspan=3)
 
    portLabel = tk.Label(loginFrame,text='Port').grid(column=1,row=2,columnspan=2,pady=10)
    portEntry = tk.Entry(loginFrame,)
    portEntry.grid(column=3,row=2,columnspan=3,pady=10)

    ipEntry.insert(0,IP)
    # ipEntry.insert(0,'192.168.43.105')
    portEntry.insert(0,str(port))

        # ipEntry.insert(0,'192.168.62.117')
        # portEntry.insert(0,'30301')

    def cert():
        '''这里需要验证用户名和密码对不对，不对就蹦出个对话框告诉他，对就destroy'''
        global IP
        global port
        IP = ipEntry.get()
        port = int(portEntry.get())
        loginFrame.destroy()#我这里为了测试直接销毁了
        print("loginFrame.destroy()")
    
    def sendCert():
        global willSend
        willSend = True
        cert()
    
    tk.Button(loginFrame,text='接收登录',command=cert).grid(column=3,row=3,padx=10,pady=15)
    tk.Button(loginFrame,text='发送登录',command=sendCert).grid(column=2,row=3,padx=50,pady=15)
    return loginFrame,willSend
                    

if __name__ == "__main__":
    #网络相关
    #收到图像的大小
    top = tk.Tk()
    top.title("MainFrame")
    loginFrame,willSend = login(top)
    try:#因为用户可能直接关闭主窗口，所以我们要捕捉这个错误
        top.wait_window(window=loginFrame) #等待直到login销毁，不销毁后面的语句就不执行
    except Exception as e:
        print(e)
        pass
    if(willSend == False):
        OBU_Recv_Video.main_thread(IP, port)
    if(willSend == True):
        OBU_Send_Video.main_thread(IP, port)
    # if(willSend == True):
    #     ikun = SendCv2(ip,port,False) #此处的True代表传输图片
    top.destroy()

