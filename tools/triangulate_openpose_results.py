'''
# 思路
# 目标: 使用3D带脚数据集训练voxelpose
1. json 2d pose 转换成 3DPose
2. 3d dataset
3. training
'''

import json
import os

import numpy as np

'''
{
    "people" : [
        {},
        {},
        ...
    ],
    1: 0
}
'''
'''
Args:
        camera_params: a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: ndarray of shape nxkx2, len(cameras) == n
    Returns:
        poses3d: ndarray of shape n/nviews x k x 3
'''



def load_pose2d_from_json(json_path):
    with open(json_path) as a:
        result = json.load(a)
        people = result['people']
        num_people = len(people)
        pose2d=[]
        for i in range(num_people):
            person = people[i]
            keypoints2d = person['pose_keypoints_2d']
            pose2d.append(keypoints2d)
        pose2d=np.array(pose2d)
        pose2d=pose2d.reshape(( -1,3))
        pose2d=pose2d[:,:2]
        return pose2d


view_folder_list=[
    "/home/tww/Datasets/output/Camera0/json",
    "/home/tww/Datasets/output/Camera1/json",
    "/home/tww/Datasets/output/Camera2/json",
]
files = os.listdir(view_folder_list[0])
for filename in files: # 对于每一zhen
    pose2d=[]
    for i in range(3):
        filename=filename[:9]+str(i)+filename[10:]
        json_path = os.path.join(view_folder_list[i], filename)
        pose2d_i=load_pose2d_from_json(json_path) # k,2
        pose2d.append(pose2d_i)




    view0 = []
    view1 = []
    view2 = []
    # pose2d=[]
    # pose2d.append(view0)
    # pose2d.append(view1)
    # pose2d.append(view2)

    pose2d = [view0, view1, view2]

