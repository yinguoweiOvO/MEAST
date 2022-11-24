import argparse
import torch
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from model import EAST
import cfg
from preprocess import resize_image
from nms import nms
from cut_test import cut

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

def load_pil(img):
    '''convert PIL Image to torch.Tensor
    '''
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
    return t(img).unsqueeze(0)

def detect(img_path, model, device, pixel_threshold, quiet=True):
    # img_dir, sub_dir, img_name = img_path.split('/')
    # print(img_name)
    img_name = img_path.split('/')[-1]
    img = Image.open(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    with torch.no_grad():
        east_detect=model(load_pil(img).to(device))
    y = np.squeeze(east_detect.cpu().numpy(), axis=0)
    y[:3, :, :] = sigmoid(y[:3, :, :])
    cond = np.greater_equal(y[0, :, :], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[1, i, j] >= cfg.side_vertex_pixel_threshold:
                if y[2, i, j] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[2, i, j] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
        # im.save(img_dir + '/predict_image/' + img_name[:-4] + '_act.jpg')
        # im.save(img_path[:-4] + '_act.jpg')
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):

            if np.amin(score) > 0:
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='red')

                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                # 转为最小矩形
                rect = cv2.minAreaRect(np.array(rescaled_geo, dtype=int))
                rescaled_geo = np.int0(cv2.boxPoints(rect))

                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item + '\n')
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        if cfg.predict_cut_text_line and len(txt_items) > 0:
            try:
                cut(txt_items, img_path)
            except Exception as e:
                with open('error.txt', 'a+') as f:
                    f.write(img_path + ': ' + str(e))
        # quad_im.save(img_dir + '/predict_image/' + img_name[:-4] + '_predict.jpg')
        quad_im.save('./predict/imgs/' + img_name[:-4] + '_predict.jpg')
        if cfg.predict_write2txt and len(txt_items) > 0:
            # with open(img_path[:-4] + '.txt', 'w') as f_txt:
            with open('./predict/txts/' + img_name[:-4] + '.txt', 'w') as f_txt:
                f_txt.writelines(txt_items)
                f_txt.close()
        else:
            # with open(img_path[:-4] + '.txt', 'w') as f_txt:
            with open('./predict/txts/' + img_name[:-4] + '.txt', 'w') as f_txt:
                f_txt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='./00001.jpg',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    # img_path = args.path
    threshold = float(args.threshold)
    model_path="/data/yinguowei/mirror_EAST/model_a-1_img-640/model/mixture_alpha-1.0/model_epoch_218_train_loss0.17093721194937825.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # for filedir, subdir, imgs_name in os.walk('D:/dataset/vin'):
    #     for img in imgs_name:
    #         img_path = os.path.join(filedir, img).replace('\\', '/')
    #         print(img_path)
    #         detect(img_path, model, device, threshold)

    # dirpath = 'test/image/'
    # for img in os.listdir(dirpath):
    #     img_path = dirpath + img
    #     print(img_path)
    #     detect(img_path, model, device, threshold)

    # detect('demo/21519.jpg', model, device, threshold)
    img_root = '/home/yinguowei/data/VIN_20000/Steel_Seal/origin/images_MBV3_640/'
    for img in os.listdir(img_root):
        detect(img_root + img, model, device, threshold)

    img_root = '/home/yinguowei/data/VIN_20000/Rubbing/origin/images_MBV3_640/'
    for img in os.listdir(img_root):
        detect(img_root + img, model, device, threshold)