import numpy as np
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import cfg
from preprocess import resize_image
import random


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, device=x.device)
    return x[tuple(indices)]


def process_flip(y):
    # temp = y[2][y[1]==1]
    # temp[temp==1] = 2
    # temp[temp==0] = 1
    # temp[temp==2] = 0
    # x = torch.zeros_like(y)
    # x[0:2] = y[0:2]
    # x[2] = y[2]
    # x[2][x[1]==1] = temp
    x = torch.zeros_like(y)
    x[0:3] = y[0:3]
    x[3] = -y[5]
    x[4] = y[6]
    x[5] = -y[3]
    x[6] = y[4]

    return x


class custom_dataset(data.Dataset):
    def __init__(self, img_path, train_val, is_mixture=False):  # train_image_dir_name
        super(custom_dataset, self).__init__()
        # 图片路径列表
        # self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]

        data_dir = cfg.data_dir
        filename = cfg.train_task_id
        if train_val == 'train':
            txt = 'train_' + filename + '.txt'
        else:
            txt = 'val_' + filename + '.txt'
        # with open(data_dir + '/' + txt) as img_txt:
        #     img_files = img_txt.readlines()
        #     self.img_files = [os.path.join(img_path, img_file.split(',')[
        #                                    0]) for img_file in img_files]
        if is_mixture == False:
            with open(data_dir + '/' + txt) as img_txt:
                img_files = img_txt.readlines()
                self.img_files = [os.path.join(img_path, img_file.split(',')[
                    0]) for img_file in img_files]
        elif is_mixture == True:
            data_dir = data_dir.replace('mixture', 'Steel_Seal')
            steel_txt = data_dir + '/' + txt
            rubbing_txt = steel_txt.replace('Steel_Seal', 'Rubbing')
            train_img_path = os.path.join(data_dir+'origin', cfg.train_image_dir_name)
            with open(steel_txt) as img_txt:
                steel_img_files = img_txt.readlines()
                steel_img_files = [train_img_path +
                                   img_file.split(',')[0] for img_file in steel_img_files]
            with open(rubbing_txt) as img_txt:
                rubbing_img_files = img_txt.readlines()
                rubbing_img_files = [train_img_path.replace(
                    'Steel_Seal', 'Rubbing')+img_file.split(',')[0] for img_file in rubbing_img_files]
                
            self.img_files = steel_img_files+rubbing_img_files
            random.shuffle(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_filename = self.img_files[index].strip()
        gt_file = img_filename.replace('images', 'labels')[:-4]+'_gt.npy'
        gt = np.load(gt_file)
        gt = torch.Tensor(gt).permute(2, 0, 1)
        img = Image.open(self.img_files[index])
        mirror_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_gt = np.load(gt_file.replace('origin', 'mirror'))
        mirror_gt = torch.Tensor(mirror_gt).permute(2, 0, 1)
        transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        return transform(img), gt, transform(mirror_img), mirror_gt


class test_dataset(data.Dataset):
    def __init__(self, img_path, is_mixture):  # train_image_dir_name
        super(test_dataset, self).__init__()

        data_dir = cfg.data_dir
        filename = cfg.train_task_id
        txt = 'val_' + filename + '.txt'
        if is_mixture == False:
            with open(data_dir + '/' + txt) as img_txt:
                img_files = img_txt.readlines()
                self.img_files = [os.path.join(img_path, img_file.split(',')[
                    0]) for img_file in img_files]
        elif is_mixture == True:
            data_dir = data_dir.replace('mixture', 'Steel_Seal')
            steel_txt = data_dir + '/' + txt
            rubbing_txt = steel_txt.replace('Steel_Seal', 'Rubbing')
            with open(steel_txt) as img_txt:
                steel_img_files = img_txt.readlines()
                steel_img_files = [img_path +
                                   img_file.split(',')[0] for img_file in steel_img_files]
            with open(rubbing_txt) as img_txt:
                rubbing_img_files = img_txt.readlines()
                rubbing_img_files = [img_path.replace(
                    'Steel_Seal', 'Rubbing')+img_file.split(',')[0] for img_file in rubbing_img_files]
            self.img_files = steel_img_files+rubbing_img_files
            random.shuffle(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_filename = self.img_files[index].strip()
        gt_file = img_filename.replace(
            cfg.train_image_dir_name[:-1], 'txt')[:-4]+'.txt'

        with open(gt_file) as gt_file:
            gt = []
            if gt_file is not None:
                for points in gt_file:
                    points = points.replace('\n', '').split(',')[:8]
                    gt.append(points)
            else:
                points = ['0', '0', '0', '0', '0', '0', '0', '0']
                gt.append(points)
        gt_file.close()
        img = Image.open(self.img_files[index])
        d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
        img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        return img_filename, transform(img), gt


def my_collate_fn(batch):
    img_filenames, imgs, label = zip(*batch)
    img = torch.zeros([len(imgs), imgs[0].shape[0],
                       imgs[0].shape[1], imgs[0].shape[2]])
    for i, temp_img in enumerate(imgs):
        img[i] = temp_img

    return list(img_filenames), img, list(label)


if __name__ == '__main__':
    # img = 'detect_data/train_img/000001.jpg'

    # # # gt_file = os.path.join(cfg.data_dir,
    # # #                        cfg.train_label_dir_name,
    # # #                        img.split('/')[-1][:-4] + '_gt.npy')
    # # # y = np.load(gt_file)

    # # img = Image.open(img)
    # # mirror_img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # # transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
    # #                                 transforms.ToTensor(),
    # #                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # # # print(y.shape)
    # # mirror_img.save('./1.jpg')
    # y = np.load('mirror_data/labels_MBV3_384/1_gt.npy')
    # y = torch.Tensor(y).permute(2, 0, 1)
    # y_mirror = np.load('mirror_data/labels_MBV3_384/000001_gt.npy')
    # y_mirror = torch.Tensor(y_mirror).permute(2, 0, 1)
    # print(y.shape)
    # print(y_mirror.shape)

    # # y = torch.cat((torch.unsqueeze(y, dim=0), torch.unsqueeze(y, dim=0)), 0)
    # # y_mirror = torch.cat((torch.unsqueeze(y_mirror, dim=0), torch.unsqueeze(y_mirror, dim=0)), 0)
    # # print(y.shape)

    # y = flip(y, 2)
    # x = process_flip(y)
    # # x[5] = -y[3]
    # print((x==y_mirror).sum())

    # # y_mirror[2:3][y_mirror[1:2]==1] = temp
    # # y_mirror = flip(y_mirror, 2)
    # # print((y[6]==y_mirror[4]).sum())
    # # temp = y_mirror[3]
    # # y_mirror[3] = -y_mirror[5]
    # # y_mirror[5] = -temp
    # # tmp = y_mirror[4]
    # # y_mirror[4] = y_mirror[6]
    # # y_mirror[6] = tmp
    # # print((y[5]==y_mirror[5]).sum())

    # # y_mirror = np.load('mirror_data/labels_MBV3_384/1.npy')
    # # y = np.load('mirror_data/labels_MBV3_384/000001.npy')
    # # print(y, y_mirror)
    # # x = torch.Tensor([11, 22, 33])
    # # x[0, 2] = -x[0, 2]
    # # print(x)

    test_img_path = os.path.join(cfg.data_dir, cfg.test_image_dir_name)
    testset = test_dataset(test_img_path)
    img_filename, img, label = testset[0]
    print(img_filename, label)
    # , collate_fn=my_collate_fn
    test_loader = data.DataLoader(testset, batch_size=3, shuffle=False,
                                  num_workers=4, drop_last=False, collate_fn=my_collate_fn)
    for i, (img_filenames, img, label) in enumerate(test_loader):
        print(img_filenames, label)
        break
