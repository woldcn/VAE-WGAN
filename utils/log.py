# Author：woldcn
# Create Time：2022/10/17 16:41
# Description：print and save to file.

from datetime import datetime


class Log:
    def __init__(self, file, cf=None):
        self.file = file
        self.str = ''
        self.print('.' * 50 + ' {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
        if cf != None:
            for arg in dir(cf):
                if arg[0] != '_':
                    self.print('{}: {}'.format(arg, getattr(cf, arg)))

    def print(self, str):
        print(str)
        self.str += str + '\n'

    def save(self):
        self.print('.' * 50 + ' {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M ")) + '.' * 50)
        with open(self.file, 'w') as f:
            print(self.str, file=f)
