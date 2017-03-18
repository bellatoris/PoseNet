import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


"""
input 으로 들어온 숫자만큼의 연속된 사진을 random 한 dataset 에서 꺼낸다.
output 으로는 연속된 사진과, 첫번째 사진에 대한 각각의 pose 를 return 한다.
"""


# dataset_train.txt로 부터 image path와 pose를 꺼내옴
def make_dataset(dir, train=True):
    paths = None
    poses = None
    # 한번에 다 담아야 함
    for target in os.listdir(dir):
        target_dir = os.path.join(dir, target)
        if not os.path.isdir(target_dir):
            continue

        # 두번 읽어야 해서 비효율 적임 창현씨랑 고민해 볼 것
        if train:
            path = np.genfromtxt(os.path.join(target_dir, 'dataset_train.txt'),
                                 dtype=np.str_, delimiter=' ', skip_header=3,
                                 usecols=[0])
            pose = np.genfromtxt(os.path.join(target_dir, 'dataset_train.txt'),
                                 dtype=np.float32, delimiter=' ', skip_header=3,
                                 usecols=[1, 2, 3, 4, 5, 6, 7])
        else:
            path = np.genfromtxt(os.path.join(target_dir, 'dataset_test.txt'),
                                 dtype=np.str_, delimiter=' ', skip_header=3,
                                 usecols=[0])
            pose = np.genfromtxt(os.path.join(target_dir, 'dataset_test.txt'),
                                 dtype=np.float32, delimiter=' ', skip_header=3,
                                 usecols=[1, 2, 3, 4, 5, 6, 7])
        # order 를 path 의 이름순으로 정한다
        order = path.argsort()

        # order 로 sorting
        path = path[order]
        pose = pose[order]

        path = np.core.defchararray.add(target + '/', path)

        if paths is None:
            paths = path
            poses = pose
        else:
            paths = np.hstack((paths, path))
            poses = np.vstack((poses, pose))

    return paths, poses


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_rotation_matrix(base_rotation):
    rotation_matrix = np.zeros((4, 4))
    rotation_matrix[0, 0] = base_rotation[0]
    rotation_matrix[0, 1] = base_rotation[3]
    rotation_matrix[0, 2] = -base_rotation[2]
    rotation_matrix[0, 3] = base_rotation[1]

    rotation_matrix[1, 0] = -base_rotation[3]
    rotation_matrix[1, 1] = base_rotation[0]
    rotation_matrix[1, 2] = base_rotation[1]
    rotation_matrix[1, 3] = base_rotation[2]

    rotation_matrix[2, 0] = base_rotation[2]
    rotation_matrix[2, 1] = -base_rotation[1]
    rotation_matrix[2, 2] = base_rotation[0]
    rotation_matrix[2, 3] = base_rotation[3]

    rotation_matrix[3, 0] = -base_rotation[1]
    rotation_matrix[3, 1] = -base_rotation[2]
    rotation_matrix[3, 2] = -base_rotation[3]
    rotation_matrix[3, 3] = base_rotation[0]

    return rotation_matrix


def make_inverse_rotation_matrix(base_rotation):
    inverse_rotation = np.copy(base_rotation)
    inverse_rotation[1:] = -inverse_rotation[1:]

    return make_rotation_matrix(inverse_rotation)


class SeqPoseData(data.Dataset):
    def __init__(self, root, seq_length=5, transform=None, target_transform=None,
                 loader=default_loader, train=True):
        paths, poses = make_dataset(root, train)

        self.root = root
        self.paths = paths
        self.poses = poses
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.seq_length = seq_length

    def __getitem__(self, index):
        target = []
        imgs = []
        base_pose = None

        # index 처리
        if self.paths[index].split("/")[1] != self.paths[index + self.seq_length].split("/")[1]:
            index -= self.seq_length

        for i in range(0, self.seq_length):
            # get relative pose
            # q_s1, q_s2 가 있을 때, q_12 = q_s1.inverse * q_s2
            pose = self.poses[index + i]
            if i == 0:
                base_trans = pose[:3]
                base_rotation = pose[3:]
                rotation_matrix = make_inverse_rotation_matrix(base_rotation)
            else:
                trans = pose[:3] - base_trans
                rotation = np.dot(rotation_matrix, pose[3:])
                pose = np.hstack((trans, rotation))
            target.append(pose)

            # img
            path = self.paths[index + i]
            img_path = os.path.join(self.root, path)
            imgs.append(self.loader(img_path))

            if self.transform is not None:
                imgs[i] = self.transform(imgs[i])
            if self.target_transform is not None:
                target[i] = self.target_transform(target[i])

            target[i] = torch.from_numpy(target[i])

        return imgs, target

    def __len__(self):
        # index 처리
        return len(self.paths) - self.seq_length

