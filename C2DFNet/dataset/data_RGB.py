import os
from PIL import Image
from PIL import ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
import random

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, depth_root, gt_root, trainsize):
        self.trainsize = trainsize

        self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depth = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.image = sorted(self.image)
        self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.image)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)), ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.image[index])
        depth = self.binary_loader(self.depth[index])
        gt = self.binary_loader(self.gts[index])

        # 不加数据增强

        image = self.img_transform(image)
        depth = self.depth_transform(depth)
        depth = torch.div(depth.float(),255.0)
        gt = self.gt_transform(gt)
        gt = np.array(gt, dtype=np.int32)
        gt[gt <= 255/2] = 0
        gt[gt > 255/2] = 1
        gt = torch.from_numpy(gt).float()
        gt = gt.unsqueeze(0)
        gt = gt.reshape(1,self.trainsize,self.trainsize)


        # name
        name = self.image[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, depth, gt,name

    def filter_files(self):
        assert len(self.image) == len(self.gts)
        depth = []
        image = []
        gts = []
        for image_path, depth_path, gt_path in zip(self.image, self.depth, self.gts):
            img = Image.open(image_path)
            dep = Image.open(depth_path)
            gt = Image.open(gt_path)
            if img.size == gt.size == dep.size:
                image.append(image_path)
                depth.append(depth_path)
                gts.append(gt_path)
            # print(len(depth))
        print("Read done")
        self.image = image
        self.depth = depth
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, depth_root, gt_root, batchsize,numworkers, trainsize, shuffle=True, pin_memory=True,iftrain = True):

    dataset = SalObjDataset(image_root, depth_root, gt_root, trainsize)
    # 获取数量
    numbers = len(dataset)
    # 多卡
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  num_workers=numworkers,
                                  shuffle=shuffle,
                                  pin_memory=pin_memory) # 多卡,sampler=train_sampler
    return data_loader,numbers


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        #　self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt, dtype=np.int32)
        gt[gt <= 255/2] = 0
        gt[gt > 255/2] = 1
        gt = torch.from_numpy(gt).float()
        gt = gt.unsqueeze(0)
        gt = gt.reshape(1,self.trainsize,self.trainsize)

        name = self.images[self.index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


