# import numpy as np
#
#
# def visualize_3d_array(array,path=None):
#     '''
#     显示3d数组
#     '''
#

import numpy as np
import matplotlib.pyplot as plt

def plot_3d_array(arr,path=None):
    # 创建一个3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 获取数组的形状
    x_len, y_len, z_len = arr.shape

    # 创建网格
    x, y, z = np.meshgrid(np.arange(x_len), np.arange(y_len), np.arange(z_len))

    # 绘制数组
    ax.scatter(x, y, z, c=arr.flatten())

    # 显示图形
    plt.show()
    # plt.savefig(path)

# 创建一个3维数组
arr = np.random.rand(80, 80, 20)

# 可视化数组
plot_3d_array(arr)

# array=np.random.random((10,10,10))
# visualize_3d_array(array)