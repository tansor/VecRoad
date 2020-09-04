import os
from multiprocessing import Pool

import cv2 as cv
import numpy as np

from lib import geom, graph as graph_helper
from utils.regions import Region, get_regions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--graph_dir", type=str, help="input predict graph dir", default="data/graphs/vecroad_4/graphs_junc/"
)
parser.add_argument(
    "--save_dir", type=str, help="save seg dir", default="data/graphs/vecroad_4/graphs_junc_seg/"
)
parser.add_argument(
    "--region_file", type=str, help="test_region.txt file path", default="data/input/regions/test_regions.txt"
)
parser.add_argument(
    "--img_size", type=int, help="generated image size", default=8192
)
parser.add_argument(
    "--thickness", type=int, help="generated road line thickness", default=8
)

args = parser.parse_args()


def worker(region):
    graph_path = os.path.join(args.graph_dir, region.name + ".graph")
    if not os.path.isfile(graph_path):
        print("graph: {} not found.".format(region.name))
        return
    g = graph_helper.read_graph(graph_path)
    img_mask = np.zeros((args.img_size, args.img_size), dtype=np.uint8)
    print(" |-> Generating {}_{}_{}".format(region.name, region.radius_y, region.radius_x))
    bound_pnt_min = geom.Point(region.radius_x, region.radius_y).scale(args.img_size // 2)
    for edge in g.edges.values():
        src = edge.src(g).point.sub(bound_pnt_min)
        dst = edge.dst(g).point.sub(bound_pnt_min)
        cv.line(img_mask, (src.x, src.y), (dst.x, dst.y), 255, args.thickness)
    # Draw mask
    cv.imwrite(os.path.join(args.save_dir, region.name + ".png"), img_mask)


if __name__ == '__main__':
    regions = get_regions(args.region_file)
    os.makedirs(args.save_dir, exist_ok=True)
    pool = Pool()  # Number of workers
    pool.map(worker, regions.values())
    pool.close()
    pool.join()
    # worker(regions['amsterdam'])
