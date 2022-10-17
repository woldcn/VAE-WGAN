# Author：woldcn
# Create Time：2022/10/17 14:53
# Description：painting functions.

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def single_line(data, save_path, label_x='', label_y='', title=''):
    plt.plot(data)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    # 刻度设置为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # 保存必须在显示之前，否则保存的是空白的
    plt.savefig(save_path)
    plt.show()
