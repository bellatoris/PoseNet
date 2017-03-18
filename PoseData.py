import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import torch


# dataset_train.txt로 부터 image path와 pose를 꺼내옴
def make_dataset(dir, train=True):
    # 두번 읽어야 해서 비효율 적임 창현씨랑 고민해 볼 것
    if train:
        paths = np.genfromtxt(os.path.join(dir, 'dataset_train.txt'),
                              dtype=str, delimiter=' ', skip_header=3,
                              usecols=[0])
        poses = np.genfromtxt(os.path.join(dir, 'dataset_train.txt'),
                              dtype=np.float32, delimiter=' ', skip_header=3,
                              usecols=[1, 2, 3, 4, 5, 6, 7])
    else:
        paths = np.genfromtxt(os.path.join(dir, 'dataset_test.txt'),
                              dtype=str, delimiter=' ', skip_header=3,
                              usecols=[0])
        poses = np.genfromtxt(os.path.join(dir, 'dataset_test.txt'),
                              dtype=np.float32, delimiter=' ', skip_header=3,
                              usecols=[1, 2, 3, 4, 5, 6, 7])

    # order를 path의 이름순으로 정한다
    order = paths.argsort()

    # order로 sorting
    paths = paths[order]
    poses = poses[order]

    return paths, poses


def default_loader(path):
    return Image.open(path).convert('RGB')


class PoseData(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, train=True):
        paths, poses = make_dataset(root, train)

        self.root = root
        self.paths = paths
        self.poses = poses
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.paths[index]
        target = self.poses[index]
        img = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.from_numpy(target)
        return img, target

    def __len__(self):
        return len(self.paths)

a = PoseData("./dataset/KingsCollege")
for i in range(0, len(a)):
    a.__getitem__(i)