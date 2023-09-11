import json
import random

import numpy as np
import scipy.io as scio

#
# dataFile = '/home/tww/Datasets/CampusSeq1/CampusSeq1/CampusSeq1/actorsGT.mat'
# data = scio.loadmat(dataFile)
# # print(data['actor3D'])
# # print(data['actor3D'].shape)
# # print(data['actor3D'][0][0].shape)
# # print(data['actor3D'][0][1].shape)
# # print(data['actor3D'][0][2].shape)
#
# actor_3d = np.array(np.array(data['actor3D'].tolist()).tolist()).squeeze()
# print(actor_3d.shape)
# print(actor_3d[0][1000].shape)
#
#
# num_actors=3
# num_views=6
# num_frames=2715
# num_keypoints=20
# # actor3D=np.array([]).reshape((num_actors, num_views, num_frames, num_keypoints, 3))
# # actor2D: 1x3 x 2000x1 x 1x3 x 14x2

'''
std::vector<std::vector<Eigen::Matrix4Xf>> LoadSkels(const std::string& filename)
{

	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "file not exist: " << filename << std::endl;
		std::abort();
	}

	int frameSize, personSize;
	fs >> frameSize;
	std::vector<std::vector<Eigen::Matrix4Xf>> skels(frameSize);
	for (int frameIdx = 0; frameIdx < frameSize; frameIdx++) {
		fs >> personSize;
		skels[frameIdx].resize(personSize, Eigen::Matrix4Xf::Zero(4, JOINT_SIZE));
		for (int pIdx = 0; pIdx < personSize; pIdx++)
			for (int i = 0; i < 4; i++)
				for (int jIdx = 0; jIdx < JOINT_SIZE; jIdx++)
					fs >> skels[frameIdx][pIdx](i, jIdx);
					// skels3d：num_frame x num_person x 4 x num_joints
	}
	fs.close();
	return skels;
}
'''


def load_skels(gt_path):
    num_joints = 21
    with open(gt_path, 'r') as f:
        frame_size = int(f.readline())
        skels = []
        # print(ra)
        for frame_idx in range(frame_size):
            line = f.readline()  # print 大法
            # print("line:",line,frame_idx)
            person_size = int(line)
            # persons = np.zeros((person_size, 3, num_joints))
            persons = np.zeros((person_size, num_joints, 4))
            for p_idx in range(person_size):
                for i in range(4):
                    line = f.readline()  # -1.17471  -1.16863  -1.17992  -1.16595  -1.13959  -1.13299   -1.0092 -0.878702 -0.947512  -1.20802  -1.33867  -1.51895  -1.42008  -1.07838  -1.00605  -1.03024  -1.27103  -1.33498  -1.44422         0         0
                    # persons[p_idx, i, :] = line.split()
                    persons[p_idx, :, i] = line.split()
                # next(f)
            persons = persons.astype(float)
            skels.append(persons.tolist())
    return skels  # [num_frame][num_person]->[num_person][num_frame]


'''你来写一下这个函数，参照c++的代码'''
'''Eigen::Matrix2Xi LoadSyncPoints(const std::string& filename) {
	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "file not exist: " << filename << std::endl;
		std::abort();
	}

	int cnt;
	fs >> cnt;
	Eigen::Matrix2Xi syncPoints(2, cnt);
	for (int i = 0; i < cnt; i++)
		fs >> syncPoints(0, i) >> syncPoints(1, i);
	fs.close();
	//[[0,300,2735],[0,1430,13127]]
	return syncPoints;
}'''


def load_sync_points(path):
    with open(path, "r") as f:
        # n=f.readline()
        next(f)
        ls = []
        for line in f:
            line = line.strip('\n')
            ls.append(line.split(' '))
        ls = np.array(ls, dtype=int)
    return ls.transpose()


def load_actor3d():
    gt_path = "/home/tww/Datasets/4d_association_dataset/dataset/seq5/gt.txt"
    sync_points_path = "/home/tww/Datasets/4d_association_dataset/dataset/seq5/sync_points.txt"
    '''
    actor_3d[person][frame]==array of shape(21,3)
    '''
    '''
           actorGT.json:
           {
           actor3D:[num_person][num_frame][num_keypoint][3]
           }
    '''

    skels3d = load_skels(gt_path)
    syncPoints = load_sync_points(sync_points_path)
    actor_3d = []
    syncIdx = 0
    frame_size = 3882

    for frameIdx in range(frame_size):
        '''
                while (frameIdx >= syncPoints(0, syncIdx))
        			syncIdx++;
        		const int gtIdx = syncPoints(1, syncIdx - 1) + int(std::round(float(frameIdx - syncPoints(0, syncIdx - 1)) *
        			(float(syncPoints(1, syncIdx) - syncPoints(1, syncIdx - 1)) / float(syncPoints(0, syncIdx) - syncPoints(0, syncIdx - 1)))));
                '''
        while (frameIdx >= syncPoints[0][syncIdx]):
            syncIdx += 1
        gt_idx = syncPoints[1][syncIdx - 1] + int(round(float(frameIdx - syncPoints[0][syncIdx - 1]) * (
                    float(syncPoints[1][syncIdx] - syncPoints[1][syncIdx - 1]) / float(
                syncPoints[0][syncIdx] - syncPoints[0][syncIdx - 1]))))
        actor_3d.append(skels3d[gt_idx])
    return actor_3d


def job_save_actor3d():
    # 把data保存为actorsGT.json
    actor_3d = load_actor3d()
    data = {
        "actor3D": actor_3d
    }
    #
    # borders=np.zeros((3,2))
    # borders[:,0]=1e+7
    # borders[:,1]=-1e+7
    # for poses in actor_3d:
    #     poses=np.array(poses)
        # print(poses.shape)
        # print(poses[0].max())
        # a=np.min(poses[:,:,:3].reshape((-1,3)),axis=0)
        # print(a)
        # borders[:,0]=np.minimum(borders[:,0],a)
        # borders[:,1]=np.maximum(borders[:,1],np.max(poses[:,:,:3].reshape((-1,3)),axis=0))
    # print(borders)

    json_str = json.dumps(data)
    with open('/home/tww/Datasets/4d_association_dataset/dataset/images/seq5/actorsGT.json', 'w') as json_file:
        json_file.write(json_str)

#不要了
def convert_pose2d():
    path = "/home/tww/Datasets/multiview_human_dataset-master/src/matrixTest.bin"
    output_path = "/home/tww/Datasets/4d_association_dataset/dataset/images/seq4/actors2D.json"
    cameras = [
        18181920,
        18181923,
        18181924,
        18307701,
        18307863,
        18307864,
    ]
    with open(path, "r") as f:
        num_frames, num_cameras = f.readline().split()
        num_frames = int(num_frames)
        num_cameras = int(num_cameras)
        data = []
        for frame_idx in range(num_frames):
            # 字典
            data_frame = {}
            for camera_idx in range(num_cameras):
                data_camera = []
                num_person = f.readline()
                num_person = int(num_person)
                for person_idx in range(num_person):
                    x_data = f.readline().strip().split()
                    y_data = f.readline().strip().split()
                    r_data = f.readline().strip().split()
                    x_data = [float(x) for x in x_data]
                    y_data = [float(x) for x in y_data]
                    r_data = [float(x) for x in r_data]

                    xy_data = list(zip(x_data, y_data))
                    data_camera.append(xy_data)
                data_frame[str(cameras[camera_idx])]=data_camera
            data.append(data_frame)
    with open(output_path, "w") as f:
        data={
            "actor2D":data
        }
        json.dump(data, f)


if __name__ == '__main__':
    job_save_actor3d()
    # convert_pose2d()
    pass
