import json
import numpy as np

# 从JSON文件加载数据
with open('/home/tww/Projects/4d_association-windows/data/shelf/calibration.json', 'r') as f:
    cameras = json.load(f)

# 循环处理每个相机
for key, value in cameras.items():
    K = np.array(value['K']).reshape((3, 3)) # 内参矩阵
    RT = np.array(value['RT']).reshape((3, 4)) # 外参矩阵
    P = np.dot(K, RT) # 投影矩阵
    print(key)
    print(P)

