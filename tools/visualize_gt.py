import json

import matplotlib.pyplot as plt
import numpy as np

LIMBS21 = np.array([0, 0, 0, 1, 2, 2, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                  1, 13, 16, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11, 12, 14, 15, 19, 17, 18, 20]).reshape((2,-1)).T.tolist()

def plot_persons(persons):

    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
    #                     top=0.95, wspace=0.05, hspace=0.15)
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    persons=np.array(persons)

    for joint in persons:
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]])
        joint[:, 0:3] = joint[:, 0:3].dot(M)
        for k in LIMBS21:
            x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
            y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
            z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
            ax.plot(x, y,z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                    markeredgewidth=1)
    plt.show()

def display():
    gt_path="/home/tww/Datasets/4d_association_dataset/dataset/images/seq4/actorsGT.json"
    with open(gt_path,'r') as f:
        data=json.load(f)
    actor3D=data['actor3D']
    for frame_data in actor3D:
        plot_persons(frame_data)
        input()


if __name__ == '__main__':
    display()