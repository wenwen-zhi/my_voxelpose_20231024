import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# Data
thresholds = [25, 50, 75, 100, 125, 150]
aps_4xrsn = [0.3011, 0.9434, 0.9834, 0.9919, 0.9974, 0.9974]
aps_hrnet = [0.0251, 0.8758, 0.9610, 0.9917, 0.9974, 0.9974]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(thresholds, aps_4xrsn, marker='o', label='4XRSN')
plt.plot(thresholds, aps_hrnet, marker='s', label='HRNet')

plt.title('不同阈值下的平均精度')
plt.xlabel('阈值 (mm)')
plt.ylabel('平均精度 (AP)')
plt.xticks(thresholds)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

plt.show()

