import torch
import torch.nn as nn
import cfg
import torch.nn.functional as F
import numpy as np
# def get_dice_loss(gt_score, pred_score):
# 	inter = torch.sum(gt_score * pred_score)
# 	union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
# 	return 1. - (2 * inter / union)


# def get_geo_loss(gt_geo, pred_geo):
# 	d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
# 	d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
# 	area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
# 	area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
# 	w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
# 	h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
# 	area_intersect = w_union * h_union
# 	area_union = area_gt + area_pred - area_intersect
# 	iou_loss_map = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
# 	angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
# 	return iou_loss_map, angle_loss_map


# class Loss(nn.Module):
# 	def __init__(self, weight_angle=10):
# 		super(Loss, self).__init__()
# 		self.weight_angle = weight_angle

# 	def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
# 		if torch.sum(gt_score) < 1:
# 			return torch.sum(pred_score + pred_geo) * 0

# 		classify_loss = get_dice_loss(gt_score, pred_score*(1-ignored_map))
# 		iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

# 		angle_loss = torch.sum(angle_loss_map*gt_score) / torch.sum(gt_score)
# 		iou_loss = torch.sum(iou_loss_map*gt_score) / torch.sum(gt_score)
# 		geo_loss = self.weight_angle * angle_loss + iou_loss
# 		print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
# 		return geo_loss + classify_loss

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, device=x.device)
    return x[tuple(indices)]


def process_flip(y):
    # temp = y[:, 2][y[:, 1]==1]
    # temp[temp==1] = 2
    # temp[temp==0] = 1
    # temp[temp==2] = 0
    # x = torch.zeros_like(y)
    # x[:, 0:2] = y[:, 0:2]
    # x[:, 2] = y[:, 2]
    # x[:, 2][x[:, 1]==1] = temp
    x = torch.zeros_like(y)
    x[:, 0:3] = y[:, 0:3]
    x[:, 3] = -y[:, 5]
    x[:, 4] = y[:, 6]
    x[:, 5] = -y[:, 3]
    x[:, 6] = y[:, 4]

    return x

def process_y(y):
    x = torch.zeros_like(y)
    x[:, 3:7] = y[:, 3:7]
    return x


def quad_loss(y_true, y_pred):
    # loss for inside_score
    logits = y_pred[:, :1, :, :]
    labels = y_true[:, :1, :, :]
    # balance positive and negative samples in an image
    beta = 1 - torch.mean(labels)
    # first apply sigmoid activation
    predicts = torch.sigmoid(logits)
    # log +epsilon for stable cal
    inside_score_loss = torch.mean(
        -1 * (beta * labels * torch.log(predicts + cfg.epsilon) +
              (1 - beta) * (1 - labels) * torch.log(1 - predicts + cfg.epsilon)))
    inside_score_loss *= cfg.lambda_inside_score_loss

    # loss for side_vertex_code
    vertex_logits = y_pred[:, 1:3, :, :]
    vertex_labels = y_true[:, 1:3, :, :]
    vertex_beta = 1 - (torch.mean(y_true[:, 1:2, :, :])
                       / (torch.mean(labels) + cfg.epsilon))
    vertex_predicts = torch.sigmoid(vertex_logits)
    pos = -1 * vertex_beta * vertex_labels * torch.log(vertex_predicts +
                                                       cfg.epsilon)
    neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * torch.log(
        1 - vertex_predicts + cfg.epsilon)

    # positive_weights = torch.cast(torch.eq(y_true[:, :, :, 0], 1), tf.float32)
    positive_weights = torch.eq(y_true[:, 0, :, :], 1).float()
    side_vertex_code_loss = \
        torch.sum(torch.sum(pos + neg, 1) * positive_weights) / (
            torch.sum(positive_weights) + cfg.epsilon)
    side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

    # loss for side_vertex_coord delta
    g_hat = y_pred[:, 3:, :, :]
    g_true = y_true[:, 3:, :, :]
    vertex_weights = torch.eq(y_true[:, 1, :, :], 1).float()
    pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)
    side_vertex_coord_loss = torch.sum(pixel_wise_smooth_l1norm) / (
        torch.sum(vertex_weights) + cfg.epsilon)
    side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss

    return inside_score_loss, side_vertex_code_loss, side_vertex_coord_loss

def loss_mirror(y_pred, mirror_y_pred, y_true):
    # # loss for inside_score
    # logits = y_pred[:, :1, :, :]
    # labels = y_true[:, :1, :, :]
    # # balance positive and negative samples in an image
    # beta = 1 - torch.mean(labels)
    # # first apply sigmoid activation
    # predicts = torch.sigmoid(logits)
    # # log +epsilon for stable cal
    # inside_score_loss = torch.mean(
    #     -1 * (beta * labels * torch.log(predicts + cfg.epsilon) +
    #           (1 - beta) * (1 - labels) * torch.log(1 - predicts + cfg.epsilon)))
    # inside_score_loss *= cfg.lambda_inside_score_loss

    # # loss for side_vertex_code
    # vertex_logits = y_pred[:, 1:3, :, :]
    # vertex_labels = y_true[:, 1:3, :, :]
    # vertex_beta = 1 - (torch.mean(y_true[:, 1:2, :, :])
    #                    / (torch.mean(labels) + cfg.epsilon))
    # vertex_predicts = torch.sigmoid(vertex_logits)
    # pos = -1 * vertex_beta * vertex_labels * torch.log(vertex_predicts +
    #                                                    cfg.epsilon)
    # neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * torch.log(
    #     1 - vertex_predicts + cfg.epsilon)

    # # positive_weights = torch.cast(torch.eq(y_true[:, :, :, 0], 1), tf.float32)
    # positive_weights = torch.eq(y_true[:, 0, :, :], 1).float()
    # side_vertex_code_loss = \
    #     torch.sum(torch.sum(pos + neg, 1) * positive_weights) / (
    #         torch.sum(positive_weights) + cfg.epsilon)
    # side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

    # loss for side_vertex_coord delta
    g_hat = y_pred[:, 3:, :, :]
    g_true = mirror_y_pred[:, 3:, :, :]
    vertex_weights = torch.eq(y_true[:, 1, :, :], 1).float()
    pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)
    side_vertex_coord_loss = torch.sum(pixel_wise_smooth_l1norm) / (
        torch.sum(vertex_weights) + cfg.epsilon)
    side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss

    return side_vertex_coord_loss


def smooth_l1_loss(prediction_tensor, target_tensor, weights):
    n_q = torch.reshape(quad_norm(target_tensor), weights.size())
    diff = prediction_tensor - target_tensor
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.lt(abs_diff, 1)
    pixel_wise_smooth_l1norm = (torch.sum(
        torch.where(abs_diff_lt_1, 0.5 * torch.pow(abs_diff, 2), abs_diff - 0.5), 1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):
    t_shape = g_true.permute(0, 2, 3, 1)
    shape = t_shape.size()  # n h w c
    delta_xy_matrix = torch.reshape(t_shape, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = torch.pow(diff, 2)
    d = torch.sum(square, 2)
    distance = torch.sqrt(d)
    distance = distance*4.0
    distance = distance + cfg.epsilon
    return torch.reshape(distance, shape[:-1])


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        return

    def forward(self, y_true, y_pred, mirror_y_true, mirror_y_pred, alpha):
        inside_score_loss, side_vertex_code_loss, side_vertex_coord_loss = quad_loss(
            y_true, y_pred)
        mirror_inside_score_loss, mirror_side_vertex_code_loss, mirror_side_vertex_coord_loss = quad_loss(
            mirror_y_true, mirror_y_pred)

        origin_loss = inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss
        mirror_loss = mirror_inside_score_loss + \
            mirror_side_vertex_code_loss + mirror_side_vertex_coord_loss

        y_temp = flip(mirror_y_pred, 3)
        y_mirror = process_flip(y_temp)
        loss = loss_mirror(y_pred, y_mirror, y_true)

        return origin_loss + mirror_loss + alpha*loss
