import os
import logging
import time
import shutil
import json

class Logger():
    def __init__(self,rootpath=r'.',timestamp=False):
        super(Logger, self).__init__()
        self.change_path(rootpath,timestamp)

    def init_logger(self):
        if self.timestamp:
            self.logpath = os.path.join(self.rootpath,time.strftime("%y_%m_%d_%H_%M", time.localtime()))
        else:
            self.logpath=self.rootpath
        print("output: "+self.logpath)
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)       
        self.txtpath=os.path.join(self.logpath,'log.txt')

    def change_path(self,rootpath,timestamp):
        self.rootpath=rootpath
        self.timestamp=timestamp
        self.init_logger()

    def log(self,logmessage):
        file = open(self.txtpath,'a')
        file.write(logmessage+'\n')
        file.close()

    def log_and_print(self,logmessage):
        self.log(logmessage)
        print(logmessage)

    def save_param(self,para_dic):
        f = open(os.path.join(self.logpath, 'param.json'), 'w')
        f.write(json.dumps(para_dic))
        f.close()

    def new_subfolder(self,foldername):
        folderpath=os.path.join(self.logpath,foldername)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
    


class Logger1():
    def __init__(self,rootpath=r'.',timestamp=False):
        super(Logger1, self).__init__()
        self.change_path(rootpath,timestamp)

    def init_logger(self):
        if self.timestamp:
            self.timestamp_folder_name = time.strftime("%y_%m_%d_%H_%M", time.localtime())
            self.logpath = os.path.join(self.rootpath,self.timestamp_folder_name)
        else:
            self.logpath=self.rootpath
        print("output: "+self.logpath)
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)       
        self.txtpath=os.path.join(self.logpath,'log.txt')

    def change_path(self,rootpath,timestamp):
        self.rootpath=rootpath
        self.timestamp=timestamp
        self.init_logger()

    def log(self,logmessage):
        file = open(self.txtpath,'a')
        file.write(logmessage+'\n')
        file.close()

    def log_and_print(self,logmessage):
        self.log(logmessage)
        print(logmessage)

    def save_param(self,para_dic):
        f = open(os.path.join(self.logpath, 'param.json'), 'w')
        f.write(json.dumps(para_dic))
        f.close()

    def new_subfolder(self,foldername):
        folderpath=os.path.join(self.logpath,foldername)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

    def get_timestamp_folder_name(self):
        return self.timestamp_folder_name if self.timestamp else None
    


    
if __name__ == '__main__':
    logger=Logger1("bhw_log", timestamp=True)
    time = logger.get_timestamp_folder_name()
    print(time)