import os
import os.path
import random

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
        if not os.path.isdir(target_dir) or target == "Street" or target == "GreatCourt":
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
        path1 = path[order]
        pose1 = pose[order]

        # reverse order 로도 sorting
        path2 = path[order[-2::-1]]
        pose2 = pose[order[-2::-1]]

        # concat
        path = np.hstack((path1, path2))
        pose = np.vstack((pose1, pose2))

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


def make_spatial_rotation_matrix(br):
    spatial_rotation_matrix = np.zeros((3, 3), dtype=np.float32)
    spatial_rotation_matrix[0, 0] = 1 - 2 * np.square(br[2]) - 2 * np.square(br[3])
    spatial_rotation_matrix[0, 1] = 2 * (br[1]*br[2] - br[3]*br[0])
    spatial_rotation_matrix[0, 2] = 2 * (br[1]*br[3] + br[2]*br[0])

    spatial_rotation_matrix[1, 0] = 2 * (br[1]*br[2] + br[3]*br[0])
    spatial_rotation_matrix[1, 1] = 1 - 2 * np.square(br[1]) - 2 * np.square(br[3])
    spatial_rotation_matrix[1, 2] = 2 * (br[2]*br[3] - br[1]*br[0])

    spatial_rotation_matrix[2, 0] = 2 * (br[1]*br[3] - br[2]*br[0])
    spatial_rotation_matrix[2, 1] = 2 * (br[2]*br[3] + br[1]*br[0])
    spatial_rotation_matrix[2, 2] = 1 - 2 * np.square(br[1]) - 2 * np.square(br[2])

    return spatial_rotation_matrix


def make_rotation_matrix(base_rotation):
    rotation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation_matrix[0, 0] = base_rotation[0]
    rotation_matrix[0, 1] = -base_rotation[1]
    rotation_matrix[0, 2] = -base_rotation[2]
    rotation_matrix[0, 3] = -base_rotation[3]

    rotation_matrix[1, 0] = base_rotation[1]
    rotation_matrix[1, 1] = base_rotation[0]
    rotation_matrix[1, 2] = -base_rotation[3]
    rotation_matrix[1, 3] = base_rotation[2]

    rotation_matrix[2, 0] = base_rotation[2]
    rotation_matrix[2, 1] = base_rotation[3]
    rotation_matrix[2, 2] = base_rotation[0]
    rotation_matrix[2, 3] = -base_rotation[1]

    rotation_matrix[3, 0] = base_rotation[3]
    rotation_matrix[3, 1] = -base_rotation[2]
    rotation_matrix[3, 2] = base_rotation[1]
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
        self.train = train

    def __getitem__(self, index):
        target = torch.zeros((self.seq_length, 7))
        imgs = torch.zeros((self.seq_length, 3, 224, 224))

        # index 처리
        while self.paths[index].split("/")[1] !=\
                self.paths[index+(self.seq_length-1)*4].split("/")[1]:
            index -= self.seq_length

        rand_index = index
        for i in range(0, self.seq_length):
            # get relative pose
            # q_s1, q_s2 가 있을 때, q_12 = q_s1.inverse * q_s2
            pose = self.poses[rand_index]

            if i == 0:
                base_trans = pose[:3]
                base_rotation = pose[3:]

                # q_ao
                inv_rot_mat = make_inverse_rotation_matrix(base_rotation)
                # r_oa
                spatial_rot_mat = make_spatial_rotation_matrix(base_rotation)
                # r_ao
                inv_spatial_rot_mat = spatial_rot_mat.transpose()
            else:
                # r_ao * (p_ob - p_oa) = p_ab at a coordinate
                trans = np.dot(inv_spatial_rot_mat, pose[:3] - base_trans)
                # q_ao * q_ob = q_ab at a coordinate
                rotation = np.dot(inv_rot_mat, pose[3:])
                pose = np.hstack((trans, rotation))

            pose = torch.from_numpy(pose)
            if self.target_transform is not None:
                pose = self.target_transform(pose)

            # get img
            path = self.paths[rand_index]
            img_path = os.path.join(self.root, path)
            img = self.loader(img_path)
            if self.transform is not None:
                img = self.transform(img)

            target[i] = pose
            imgs[i] = img

            if self.train:
                rand_index = rand_index + random.randrange(1, 5)
            else:
                rand_index += 1

        return imgs, target

    def __len__(self):
        # index 처리
        return len(self.paths) - (self.seq_length - 1) * 4

