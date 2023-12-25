import json

import matplotlib.pyplot as plt
import numpy as np

LIMBS21 = np.array([0, 0, 0, 1, 2, 2, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                  1, 13, 16, 2, 3, 5, 9, 4, 6, 7, 8, 10, 11, 12, 14, 15, 19, 17, 18, 20]).reshape((2,-1)).T.tolist()

LIMBS23=np.array([0,0,0,1,1,1,3,4,6,7,9,9,9,9,9,14,16,16,16,16,16,21,
                  1,15,22,2,5,8,4,5,7,8,10,11,12,13,14,15,17,18,19,20,21,22]).reshape((-1, 2)).tolist()

LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]

def plot_persons(persons,save_path=None):

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
        for k in LIMBS14:
            x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
            y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
            z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
            ax.plot(x, y,z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                    markeredgewidth=1)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def display():
    # gt_path="/home/tww/Datasets/4d_association_dataset/dataset/images/seq4/actorsGT.json"
    gt_path="/home/tww/Datasets/ue/train/actorsGT.json"
    save_dir = "/home/tww/Datasets/ue/train/gt/"
    with open(gt_path,'r') as f:
        data=json.load(f)
    actor3D=data['actor3D']
    for frame_data in actor3D:
        plot_persons(frame_data,save_path=None)
        input()


if __name__ == '__main__':
    display()