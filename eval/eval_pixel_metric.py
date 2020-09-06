#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
from multiprocessing import Pool
from os.path import basename

import numpy as np
import numba
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, default="data/input/mask_test")
parser.add_argument('--pred_dir', type=str, default="data/graphs/vecroad_4/graphs_junc_seg")
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--relax', type=int, default=3)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--thresh', type=int, default=-1, help="threshold")
parser.add_argument('--crop', type=int, default=0, help="crop image boundary")
args = parser.parse_args()
print(args)

result_dir = args.pred_dir
label_dir = args.gt_dir
result_fns = sorted(glob.glob('%s/*.png' % result_dir))
n_results = len(result_fns)
if args.thresh != -1:
    args.steps = 1


def get_pre_rec(positive, prec_tp, true, recall_tp, steps):
    pre_rec = []
    breakeven = []

    for t in range(steps):
        if positive[t] < prec_tp[t] or true[t] < recall_tp[t]:
            sys.exit('calculation is wrong')
        pre = float(prec_tp[t]) / positive[t] if positive[t] > 0 else 0
        rec = float(recall_tp[t]) / true[t] if true[t] > 0 else 0
        pre_rec.append([pre, rec])
        if pre != 1 and rec != 1 and pre > 0 and rec > 0:
            breakeven.append([pre, rec])

    pre_rec = np.asarray(pre_rec)

    breakeven = np.asarray(breakeven)
    breakeven_pt = np.abs(breakeven[:, 0] - breakeven[:, 1]).argmin()
    breakeven_pt = breakeven[breakeven_pt]

    return pre_rec, breakeven_pt


def worker(img_idx, result_fn):
    img_id = basename(result_fn).split('.')[0]

    print('[%4d] %s/%s.png' % (img_idx, label_dir, img_id))

    label = cv.imread('%s/%s.png' % (label_dir, img_id), cv.IMREAD_GRAYSCALE)
    if args.crop != 0:
        label = label[args.crop:-args.crop, args.crop:-args.crop]
    label = label/255

    pred = cv.imread('%s/%s.png' % (result_dir, img_id), cv.IMREAD_GRAYSCALE)
    if args.crop != 0:
        pred = pred[args.crop:-args.crop, args.crop:-args.crop]
    pred = pred/255

    if args.thresh != -1:
        rng = [args.thresh]
    else:
        rng = np.linspace(0, 255, args.steps)

    positive = []
    prec_tp = []
    true = []
    recall_tp = []
    for thresh in rng:
        thresh = thresh * 1. / 255
        pred_vals = np.array(pred[:, :] >= thresh, dtype=np.int32)
        label_vals = np.array(label, dtype=np.int32)

        positive.append(np.sum(pred_vals))
        prec_tp.append(relax_precision(pred_vals, label_vals, args.relax))
        true.append(np.sum(label_vals))
        recall_tp.append(relax_recall(pred_vals, label_vals, args.relax))

    print('thread finished')
    return positive, prec_tp, true, recall_tp


@numba.jit
def relax_precision(predict, label, relax):
    h_lim = predict.shape[1]
    w_lim = predict.shape[0]

    true_positive = 0

    for y in range(h_lim):
        for x in range(w_lim):
            if predict[y, x] == 1:
                st_y = y - relax if y - relax >= 0 else 0
                en_y = y + relax if y + relax < h_lim else h_lim - 1
                st_x = x - relax if x - relax >= 0 else 0
                en_x = x + relax if x + relax < w_lim else w_lim - 1

                sum = 0
                for yy in range(st_y, en_y+1):
                    for xx in range(st_x, en_x+1):
                        sum += label[yy, xx]
                if sum > 0:
                    true_positive += 1

    return true_positive


@numba.jit
def relax_recall(predict, label, relax):
    h_lim = predict.shape[1]
    w_lim = predict.shape[0]

    true_positive = 0

    for y in range(h_lim):
        for x in range(w_lim):
            if label[y, x] == 1:
                st_y = y - relax if y - relax >= 0 else 0
                en_y = y + relax if y + relax < h_lim else h_lim - 1
                st_x = x - relax if x - relax >= 0 else 0
                en_x = x + relax if x + relax < w_lim else w_lim - 1

                sum = 0
                for yy in range(st_y, en_y+1):
                    for xx in range(st_x, en_x+1):
                        sum += predict[yy, xx]
                if sum > 0:
                    true_positive += 1

    return true_positive

def get_f1(pre, rec):
    if pre == rec == 0:
        return 0
    return 2 * pre * rec / (pre + rec)


if __name__ == '__main__':
    pool = Pool(args.num_workers)
    tmp_lst = []
    for i in range(n_results):
        tmp_lst.append(pool.apply_async(worker, args=(i, result_fns[i],)))
    for i in range(n_results):
        tmp_lst[i] = tmp_lst[i].get()
    print('all finished')

    all_positive = np.array([x[0] for x in tmp_lst])
    all_prec_tp = np.array([x[1] for x in tmp_lst])
    all_true = np.array([x[2] for x in tmp_lst])
    all_recall_tp = np.array([x[3] for x in tmp_lst])

    all_positive = np.sum(all_positive, axis=0)
    all_prec_tp = np.sum(all_prec_tp, axis=0)
    all_true = np.sum(all_true, axis=0)
    all_recall_tp = np.sum(all_recall_tp, axis=0)

    pre_rec, breakeven_pt = get_pre_rec(
        all_positive, all_prec_tp,
        all_true, all_recall_tp, args.steps)
    F1 = []
    for i in range(args.steps):
        if pre_rec[i, 0] == 0.:
            continue
        F1.append(get_f1(pre_rec[i,0], pre_rec[i,1]))
    
    print("BreakEven Point:")
    print("precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(
        breakeven_pt[0] * 100, breakeven_pt[1] * 100, get_f1(*breakeven_pt) * 100))
    print("max P-F1: {:.2f}".format(np.max(F1) * 100))
