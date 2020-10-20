import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root, is_train):
    if is_train:

        input = open(os.path.join(root, 'data/train_input.txt'))
        ground_t = open(os.path.join(root, 'data/train_gt.txt'))
        depth_t = open(os.path.join(root, 'data/train_depth.txt'))
        image = [(os.path.join(root, 'train', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'image', img_name.strip('\n'))) for img_name in
                 ground_t]
        depth = [(os.path.join(root, 'depth', img_name.strip('\n'))) for img_name in
              depth_t]

        input.close()
        ground_t.close()
        depth_t.close()


        return [[image[i], gt[i], depth[i]]for i in range(len(image))]

    else:

        input = open(os.path.join(root, 'data/test_input.txt'))
        ground_t = open(os.path.join(root, 'data/test_gt.txt'))
        depth_t = open(os.path.join(root, 'data/test_depth.txt'))

        image = [(os.path.join(root, 'test', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'image', img_name.strip('\n'))) for img_name in
              ground_t]
        depth = [(os.path.join(root, 'depth', img_name.strip('\n'))) for img_name in
                 depth_t]

        input.close()
        ground_t.close()
        depth_t.close()

        return [[image[i], gt[i], depth[i]]for i in range(len(image))]



class ImageFolder(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True):
        self.root = root
        self.imgs = make_dataset(root, is_train)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, depth_path = self.imgs[index]
        #print(img_path)
        #print(gt_path)
        #print(depth_path)
        img = Image.open(img_path)
        target = Image.open(gt_path)
        depth = Image.open(depth_path)
        if self.triple_transform is not None:
            img, target, depth = self.triple_transform(img, target, depth)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)

        return img, target, depth

    def __len__(self):
        return len(self.imgs)
