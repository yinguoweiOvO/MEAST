import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from generator import custom_dataset, test_dataset, my_collate_fn
from model import EAST
from loss import Loss
import os
import time
import numpy as np
import cfg
from tqdm import tqdm
from nms import nms
import shapely
from shapely.geometry import Polygon, MultiPoint
from PIL import Image, ImageDraw
from preprocess import resize_image
from cv2 import cv2
import argparse


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def bbox_iou_eval(box1, box2):
    '''
    利用python的库函数实现非矩形的IoU计算
    :param box1: list,检测框的四个坐标[x1,y1,x2,y2,x3,y3,x4,y4]
    :param box2: lsit,检测框的四个坐标[x1,y1,x2,y2,x3,y3,x4,y4]
    :return: IoU
    '''
    box1 = np.array(box1).reshape(4, 2)  # 四边形二维坐标表示
    # python四边形对象，会自动计算四个点，并将四个点从新排列成
    # 左上，左下，右下，右上，左上（没错左上排了两遍）
    poly1 = Polygon(box1).convex_hull
    box2 = np.array(box2).reshape(4, 2)
    poly2 = Polygon(box2).convex_hull

    if not poly1.intersects(poly2):  # 若是两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2)  # 相交面积
            # inter_points = inter_area.boundary
            # cv2.fillConvexPoly(img, np.array(inter_points, dtype=int), (0, 0, 255))
            iou = float(inter_area.area) / \
                (poly1.area + poly2.area - inter_area.area)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

    return iou


def train(train_img_path, batch_size, lr, decay, num_workers, epoch_iter, pretained, start_epoch, alpha, is_mixture=False):
    file_num = len(os.listdir(train_img_path))
    if is_mixture == True:
        file_num *= 2
    val_file_num = file_num * 0.2
    train_file_num = file_num - val_file_num
    trainset = custom_dataset(train_img_path, 'train', is_mixture)
    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True)

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    # TODO 可能是bug
    # if os.path.exists(pretained):
    #     print('loading model...')
    #     model.load_state_dict(torch.load(pretained))
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(start_epoch, epoch_iter):
        print('Epoch: ', epoch)
        print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        model.train()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_map, mirror_img, mirror_gt_map) in enumerate(tqdm(train_loader)):
            start_time = time.time()
            img, gt_map, mirror_img, mirror_gt_map = img.to(device), gt_map.to(
                device), mirror_img.to(device), mirror_gt_map.to(device)
            detect = model(img)
            mirror_detect = model(mirror_img)
            loss = criterion(gt_map, detect, mirror_gt_map,
                             mirror_detect, alpha)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
            # print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
#   epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))

        print('Epoch is [{}/{}], epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch+1, epoch_iter,
                                                                                    epoch_loss/int(train_file_num/batch_size), time.time()-epoch_time))

        print(time.asctime(time.localtime(time.time())))
        state_dict = model.module.state_dict() if data_parallel else model.state_dict()
        torch.save(state_dict, os.path.join('/data/yinguowei/saved_model/vin/mirror_EAST/model_a-1_img-640/model/'+cfg.data_dir.split('/')[-2]+'_alpha-'+str(args.alpha)+'/model_epoch_{}_train_loss{}.pth'.format(
            str(epoch+1).zfill(3), epoch_loss/int(train_file_num/batch_size))))


if __name__ == '__main__':
    train_img_path = os.path.join(cfg.data_dir+'origin', cfg.train_image_dir_name)
    pretained = ''
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()
    start_epoch = 0
    print('alpha:', args.alpha)
    print(train_img_path)
    if not os.path.exists('/data/yinguowei/saved_model/vin/mirror_EAST/model_a-1_img-640/model/'+cfg.data_dir.split('/')[-2]+'_alpha-'+str(args.alpha)):
        os.makedirs('/data/yinguowei/saved_model/vin/mirror_EAST/model_a-1_img-640/model/'+cfg.data_dir.split('/')[-2]+'_alpha-'+str(args.alpha))

    if cfg.data_dir.split('/')[-2] == 'mixture':
        data_dir = cfg.data_dir.replace('mixture', 'Steel_Seal')
        train_img_path = os.path.join(data_dir+'origin', cfg.train_image_dir_name)
        is_mixture = True
    else:
        is_mixture = False
    print(is_mixture)
    train(train_img_path, 8, cfg.lr, cfg.decay,
          cfg.num_workers, cfg.epoch_num, pretained, start_epoch, args.alpha, is_mixture)