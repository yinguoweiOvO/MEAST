# coding=utf-8
import numpy as np

import cfg


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    # 判断集合元素是否相等,返回true ,  not true =false
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))  # 由{(a,b)}转换为[[a b]]
    j_min = np.amin(region_pixels, axis=0)[1] - 1  # 取行最小值 ,如不指定，则是所有元素的最大值
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):  # len(region_list) =36
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)  # 方法用于在列表末尾添加新的对象
    for d in tmp:
        S.remove(d)  # 指定删除list
    for e in tmp:
        # 于在列表末尾一次性追加另一个序列中的多个值
        rows.extend(rec_region_merge(region_list, e, S))

    return rows


def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                # print(region_list)
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    quad_list = np.zeros((2, 4, 2))
    score_list = np.zeros((2, 4))
    D = region_group(region_list)
    # print(D)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[1, ij[0], ij[1]]
                if score >= threshold:  # threshold == 0.9
                    ith_score = predict[2:3, ij[0], ij[1]]
                    if not (cfg.trunc_threshold <= ith_score < 1 -
                            cfg.trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[3:7, ij[0], ij[1]],
                                                    (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    # print("score_list: ", score_list)
    # print("quad_list: ", quad_list)
    return score_list, quad_list
