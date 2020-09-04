#!/user/bin/python
# coding=utf-8

import sys
import os
import time
import math
from utils.utils import AverageMeter
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--graph_dir", type=str, help="input graph dir", default="data/graphs/vecroad_4/graphs_junc/"
)
parser.add_argument(
    "--gt_dir", type=str, help="gt graph dir", default="data/input/graphs/"
)
parser.add_argument(
    "--save_dir", type=str, help="save csv dir", default="data/graphs/vecroad_4/"
)
parser.add_argument(
    "--file_name", type=str, help="save file name", default="graphs_junc_jf1"
)

args = parser.parse_args()


def worker(fn):
    if ' ' in fn:
        tmp = fn.split(' ')
        fn = ''
        for i in range(len(tmp)-1):
            fn += tmp[i] + '\ '
        fn += tmp[-1]

    cmd = "go run eval/junction_metric.go {}.graph {} {}".format(
        os.path.join(args.gt_dir, fn),
        os.path.join(args.graph_dir, fn+'.graph'),
        fn)
    ret = os.popen(cmd).readlines()
    res = ret[-1].split(' ;;; ')
    total = float(res[0].split(' ')[0])
    correct = float(res[1].split(' ')[0])
    error = float(res[1].split(' ')[-2])
    print(fn, total, correct, error)
    return [total, correct, error]


if __name__ == '__main__':
    files = os.listdir(args.graph_dir)
    files.sort()
    files = [fn.split('.')[0] for fn in files]
    csv_path = os.path.join(args.save_dir, args.file_name + '.csv')
    pool = Pool()
    ret_dict = {}
    for fn in files:
        ret = pool.apply_async(worker, args=(fn,))
        ret_dict[fn] = ret
    for k, v in ret_dict.items():
        ret_dict[k] = v.get()
    pool.close()
    pool.join()
    csv_file = open(csv_path, 'a')
    total_recall = AverageMeter()
    total_precision = AverageMeter()
    total_f1 = AverageMeter()
    for fn in files:
        total, correct, error = ret_dict[fn]
        recall = correct / total
        if correct + error == 0:
            precision = 0
        else:
            precision = correct/(correct+error)
        f1 = 2 * precision * recall / (precision + recall)
        total_recall.update(recall)
        total_precision.update(precision)
        total_f1.update(f1)
        csv_file.write('{},{},{},{},{},{},{}\n'.format(fn, total, correct, error, precision, recall, f1))
    csv_file.write('{},,,,{},{},{}\n\n'.format(args.graph_dir.split('/')[-1], total_precision.avg, total_recall.avg,  total_f1.avg))
    csv_file.close()
