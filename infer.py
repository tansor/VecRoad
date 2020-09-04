import argparse
import math
import os.path
import random
import sys
import time
from multiprocessing import Pool

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict
from PIL import Image
from skimage import measure
from skimage.morphology import thin
from tqdm import tqdm

import utils.model_utils as model_utils
import utils.tileloader as tileloader
from lib import geom, graph as graph_helper
from model.model import RPNet, upsample
from utils.regions import Region, get_regions
from utils.utils import load_pretrained, numpy2tensor2cuda, MapContainer

parser = argparse.ArgumentParser(description="VecRoad Pytorch Test")
parser.add_argument(
    "--config",
    default="configs/default.yml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()

assert os.path.isfile(args.config)
config_file = open(args.config, "r")
cfg = yaml.load(config_file, Loader=yaml.UnsafeLoader)
config_file.close()
cfg = EasyDict(cfg)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TEST.GPU_ID


def main():
    test_regions = get_regions(cfg.DIR.TEST_REGION_PATH)
    if cfg.TEST.SINGLE_REGION != "":
        test_regions = {
            cfg.TEST.SINGLE_REGION: test_regions[cfg.TEST.SINGLE_REGION]}

    net = prepare_net().eval()

    junction_nms_res = dict()
    road_seg_filter_dict = dict()
    graph_dict = dict()
    for region_name in test_regions.keys():
        graph_dict[region_name] = None
    if cfg.TEST.INFER_STEP == "start":
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junction"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "road"), exist_ok=True)
        print("infer segmentation start, INPUT_SIZE:{}".format(cfg.TEST.CROP_SZ))
        road_map_dict, junc_map_dict = infer_segmentation(
            net, list(test_regions.keys()))
        print("infer segmentation done")
        print("junction nms start")
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms_vis"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                                 "road_seg_region_filter"), exist_ok=True)
        pool = Pool(cfg.TEST.CPU_WORKER)
        if cfg.TEST.START_FROM_JUNC_PEAK:
            for region_name in test_regions.keys():
                junction_nms_res[region_name] = pool.apply_async(
                    junction_nms, args=(region_name, junc_map_dict[region_name],))
            for region_name in test_regions.keys():
                junction_nms_res[region_name] = junction_nms_res[region_name].get()
        del junc_map_dict
        if cfg.TEST.START_FROM_ROAD_PEAK:
            for region_name in test_regions.keys():
                road_seg_filter_dict[region_name] = pool.apply_async(
                    road_seg_region_filter, args=(region_name, road_map_dict[region_name],))
            for region_name in test_regions.keys():
                road_seg_filter_dict[region_name] = road_seg_filter_dict[region_name].get(
                )
        del road_map_dict
        pool.close()
        pool.join()
        print("junction nms done")
    elif cfg.TEST.INFER_STEP == "after_seg":
        print("junction nms start")
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR,
                                 cfg.TEST.CKPT, "junc_nms_vis"), exist_ok=True)
        os.makedirs(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                                 "road_seg_region_filter"), exist_ok=True)
        pool = Pool(cfg.TEST.CPU_WORKER)
        if cfg.TEST.START_FROM_JUNC_PEAK:
            for region_name in test_regions.keys():
                junc_nms_map = cv.imread(os.path.join(
                    cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "junction", region_name + '.png'), 0)
                junction_nms_res[region_name] = pool.apply_async(
                    junction_nms, args=(region_name, junc_nms_map,))
            for region_name in test_regions.keys():
                junction_nms_res[region_name] = junction_nms_res[region_name].get()
        if cfg.TEST.START_FROM_ROAD_PEAK:
            for region_name in test_regions.keys():
                road_seg_map = cv.imread(os.path.join(
                    cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road", region_name + '.png'), 0) / 255.
                road_seg_filter_dict[region_name] = pool.apply_async(
                    road_seg_region_filter, args=(region_name, road_seg_map,))
            for region_name in test_regions.keys():
                road_seg_filter_dict[region_name] = road_seg_filter_dict[region_name].get(
                )
        pool.close()
        pool.join()
        print("junction nms end")
    elif cfg.TEST.INFER_STEP in ["after_junc_nms", "given_junc_nms"] and cfg.TEST.START_FROM_ROAD_PEAK:
        for region_name in test_regions.keys():
            road_seg_filter_dict[region_name] = cv.imread(
                os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road_seg_region_filter", region_name + ".png"), 0) / 255.
    elif cfg.TEST.INFER_STEP == "after_graph_from_junc" and cfg.TEST.START_FROM_ROAD_PEAK:
        for region_name in test_regions.keys():
            road_seg_filter_dict[region_name] = cv.imread(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                                                                       "road_seg_region_filter", region_name + ".png"), 0) / 255.
            if cfg.TEST.START_FROM_JUNC_PEAK:
                graph_dict[region_name] = graph_helper.read_graph(os.path.join(
                    cfg.DIR.SAVE_GRAPH_DIR, '{}_{}'.format(
                        cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS), 'graphs_junc',
                    '{}.graph'.format(region_name)))

    img_cache = tileloader.TileCache(
        tile_dir=cfg.DIR.IMAGERY_DIR,
        tile_size=cfg.TRAIN.IMG_SZ,
        window_size=cfg.TEST.WINDOW_SIZE,
        limit=cfg.TRAIN.PARALLEL_TILES)
    paths = []
    region_lst = list(test_regions.keys())

    if not cfg.TEST.INFER_STEP == "after_graph_from_junc" and cfg.TEST.START_FROM_JUNC_PEAK:
        for i, region_name in enumerate(region_lst):
            tile_data = get_tile_data(
                test_regions[region_name], img_cache, junction_nms_res, get_starting_locations=True)
            paths.append(model_utils.Path(i, training=False, gc=None, tile_data=tile_data,
                                          graph=None, road_seg=None))

        save_graph_dir = os.path.join(cfg.DIR.SAVE_GRAPH_DIR, '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
                                      'graphs_junc')
        os.makedirs(save_graph_dir, exist_ok=True)
        try:
            iters, graph_dict = infer_anchor(paths, net, region_lst=region_lst, save_graph_dir=save_graph_dir,
                                             batch_size=cfg.TEST.BATCH_SIZE_ANCHOR)
            print(iters)
        except:
            for path in paths:
                path.graph.save(os.path.join(
                    save_graph_dir, 'except_{}.graph'.format(region_lst[path.idx])))
                print("    Except save graph {}".format(region_lst[path.idx]))

    if cfg.TEST.START_FROM_ROAD_PEAK:
        if len(paths) == 0:
            for i, region_name in enumerate(region_lst):
                tile_data = get_tile_data(
                    test_regions[region_name], img_cache, junction_nms_res, get_starting_locations=False)
                path = model_utils.Path(i, training=False, gc=None, tile_data=tile_data,
                                        graph=graph_dict[region_name],
                                        road_seg=np.ascontiguousarray(road_seg_filter_dict[region_name].swapaxes(0, 1)))
                paths.append(path)
        else:
            for i, region_name in enumerate(region_lst):
                paths[i].road_seg = np.ascontiguousarray(
                    road_seg_filter_dict[region_name].swapaxes(0, 1))
                paths[i].remove_graph_from_road_seg()

        if cfg.TEST.START_FROM_JUNC_PEAK:
            save_graph_dir = os.path.join(cfg.DIR.SAVE_GRAPH_DIR,
                                          '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
                                          'graphs_junc_road')
        else:
            save_graph_dir = os.path.join(cfg.DIR.SAVE_GRAPH_DIR,
                                          '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
                                          'graphs_road')
        os.makedirs(save_graph_dir, exist_ok=True)
        try:
            iters, graph_dict = infer_anchor(paths, net, region_lst=region_lst, save_graph_dir=save_graph_dir,
                                             batch_size=cfg.TEST.BATCH_SIZE_ANCHOR)
            print(iters)
        except:
            for path in paths:
                path.graph.save(os.path.join(
                    save_graph_dir, 'except_{}.graph'.format(region_lst[path.idx])))
                print("    Except save graph {}".format(region_lst[path.idx]))

    post_process_graph(graph_dict)


def infer_anchor(paths, net, region_lst, save_graph_dir, batch_size=15, save_pic=True,
                 max_iteration=99999999, verbose=True):
    print("infer anchor start")
    net.eval()
    if len(paths) >= batch_size:
        pass
    else:
        batch_size = len(paths)
    print("batch_size:" + str(batch_size))
    output_flag_list = [False for _ in range(len(paths))]
    graph_dict = dict()

    iteration = 0

    for iteration in range(max_iteration):
        path_indices = []
        batch_extension_vertices = []
        batch_is_key_point = np.empty(batch_size)
        batch_inputs = np.empty(
            (batch_size, 3, cfg.TEST.WINDOW_SIZE, cfg.TEST.WINDOW_SIZE))
        batch_walked_path = np.empty(
            (batch_size, 1, cfg.TEST.WINDOW_SIZE // 4, cfg.TEST.WINDOW_SIZE // 4))

        for path_idx in range(len(paths)):
            if output_flag_list[path_idx]:
                continue

            extension_vertex, is_key_point = paths[path_idx].pop(
                follow_order=True)
            if extension_vertex is None:
                output_flag_list[path_idx] = True
                paths[path_idx].graph.save(os.path.join(
                    save_graph_dir,
                    '{}.graph'.format(region_lst[path_idx])))
                print("    save graph {}".format(region_lst[path_idx]))
                graph_dict[region_lst[path_idx]] = paths[path_idx].graph
                continue
            i = len(path_indices)
            path_indices.append(path_idx)
            batch_extension_vertices.append(extension_vertex)
            batch_is_key_point[i] = is_key_point
            fetch_list = ['aerial_image_chw', 'walked_path_small']
            if cfg.TEST.SAVE_EXAMPLES:
                fetch_list += ['aerial_image_hwc']
            data_dict = paths[path_idx].make_path_input(
                extension_vertex=extension_vertex,
                fetch_list=fetch_list,
                is_key_point=is_key_point,
                WINDOW_SIZE=cfg.TEST.WINDOW_SIZE)
            data_dict = EasyDict(data_dict)
            batch_inputs[i] = data_dict.aerial_image_chw
            batch_walked_path[i] = data_dict.walked_path_small
            if len(path_indices) >= batch_size:
                break

        if len(path_indices) == 0:
            break
        length_path_indices = len(path_indices)
        batch_is_key_point = batch_is_key_point[:length_path_indices]
        batch_inputs = batch_inputs[:length_path_indices]
        batch_walked_path = batch_walked_path[:length_path_indices]

        batch_inputs_cuda = numpy2tensor2cuda(batch_inputs)
        batch_walked_path_cuda = numpy2tensor2cuda(batch_walked_path)

        # network infer
        batch_output_cuda_dict = net(
            batch_inputs_cuda, batch_walked_path_cuda, NUM_TARGETS=cfg.TEST.NUM_TARGETS)

        batch_output_road_cuda = batch_output_cuda_dict['road']
        batch_output_junc_cuda = batch_output_cuda_dict['junc']
        batch_output_anchor_maps_cuda = batch_output_cuda_dict['anchor']

        batch_output_road_cuda = upsample(batch_output_road_cuda, 4)
        batch_output_road = torch.sigmoid(
            batch_output_road_cuda).detach().cpu().numpy()

        batch_output_anchor_maps = torch.sigmoid(
            batch_output_anchor_maps_cuda).detach().cpu().numpy()

        if cfg.TEST.SAVE_EXAMPLES and cfg.TEST.START_FROM_JUNC_PEAK:
            batch_output_junc = torch.sigmoid(
                batch_output_junc_cuda).detach().cpu().numpy()

        batch_output_points = model_utils.map_to_coordinate(
            batch_output_maps=batch_output_anchor_maps.copy(),
            batch_is_key_point=batch_is_key_point,
            batch_extension_vertices=batch_extension_vertices,
            ROAD_SEG_THRESHOLE=cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE,
            STEP_LENGTH=cfg.TEST.STEP_LENGTH,
            JUNC_MAX_REGION_AREA=cfg.TEST.BINARIZE_MAP.JUNC_MAX_REGION_AREA)

        if verbose and iteration % cfg.TEST.PRINT_ITERATION == 0:
            print('  iter:{} len(paths):{}'.format(
                iteration, len(path_indices)))

        save_idx = cfg.TEST.SAVE_IDX
        if cfg.TEST.SAVE_EXAMPLES and save_idx in path_indices:
            for i in range(len(path_indices)):
                region_name = region_lst[path_indices[i]]
                os.makedirs(os.path.join(cfg.DIR.INFER_STEP_DIR,
                                         region_name), exist_ok=True)
                fname = os.path.join(cfg.DIR.INFER_STEP_DIR,
                                     region_name, '{}_'.format(iteration))
                pred_gt_pair_list = [
                    ("anchor", batch_output_anchor_maps[save_idx], None)]
                pred_gt_pair_list.append(
                    ("road", batch_output_road[save_idx, 0], None))
                pred_gt_pair_list.append(
                    ("junc", batch_output_junc[save_idx, 0], None))
                paths[path_indices[save_idx]].visualize_output(
                    fname_prefix=fname,
                    extension_vertex=batch_extension_vertices[save_idx],
                    aerial_image=data_dict.aerial_image_hwc, target_poses=None,
                    pred_gt_pair_list=pred_gt_pair_list)

        for i in range(len(path_indices)):
            path_idx = path_indices[i]
            if len(batch_output_points[i]) > 0:
                # extension_vertex has not been added into graph
                if hasattr(batch_extension_vertices[i], 'from_road_seg'):
                    batch_extension_vertices[i] = paths[path_idx].graph.add_vertex(
                        batch_extension_vertices[i].point)
                paths[path_idx].push(
                    extension_vertex=batch_extension_vertices[i],
                    is_key_point=batch_is_key_point[i],
                    follow_mode=cfg.TEST.FOLLOW_MODE,
                    target_poses=None,
                    output_points=batch_output_points[i],
                    RECT_RADIUS=cfg.TEST.RECT_RADIUS,
                    road_segmentation=batch_output_road[i, 0],
                    NUM_TARGETS=cfg.TEST.NUM_TARGETS,
                    WINDOW_SIZE=cfg.TEST.WINDOW_SIZE,
                    STEP_LENGTH=cfg.TEST.STEP_LENGTH,
                    AVG_CONFIDENCE_THRESHOLD=cfg.TEST.AVG_CONFIDENCE_THRESHOLD)

    return iteration, graph_dict


def get_tile_data(region, cache, junction_nms_res=None, get_starting_locations=True):
    print('  region: {}'.format(region.name))
    TILE_START = geom.Point(
        region.radius_x, region.radius_y).scale(cfg.TRAIN.IMG_SZ)
    TILE_END = TILE_START.add(geom.Point(2, 2).scale(cfg.TRAIN.IMG_SZ))
    search_rect = geom.Rectangle(TILE_START, TILE_END)
    starting_locations = []

    if get_starting_locations:
        pnts = list()
        if cfg.TEST.INFER_STEP == "given_junc_nms":
            for x in range(region.radius_x, region.radius_x + 2):
                for y in range(region.radius_y, region.radius_y + 2):
                    fname = '{}_{}_{}.png'.format(region.name, x, y)
                    junc_nms_map = cv.imread(os.path.join(
                        cfg.DIR.PRE_JUNC_NMS_DIR, fname), 0)
                    tmp_pnts = list(zip(*np.where(junc_nms_map > 0)))
                    tmp_pnts = [geom.Point(pnt[1] + x * cfg.TRAIN.IMG_SZ, pnt[0] + y * cfg.TRAIN.IMG_SZ)
                                for pnt in tmp_pnts]
                    pnts.extend(tmp_pnts)
        elif cfg.TEST.INFER_STEP == "after_junc_nms":
            junc_nms_map = cv.imread(os.path.join(
                cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "junc_nms", region.name + '.png'), 0).astype(np.float32)
            pnts = list(zip(*np.where(junc_nms_map > 0)))
            pnts = [geom.Point(pnt[1] + region.radius_x * cfg.TRAIN.IMG_SZ,
                               pnt[0] + region.radius_y * cfg.TRAIN.IMG_SZ)
                    for pnt in pnts]
        elif cfg.TEST.INFER_STEP == "start" or cfg.TEST.INFER_STEP == "after_seg":
            pnts = [geom.Point(pnt[1] + region.radius_x * cfg.TRAIN.IMG_SZ, pnt[0] + region.radius_y * cfg.TRAIN.IMG_SZ)
                    for pnt in junction_nms_res[region.name]]

        for pnt in pnts:
            if not search_rect.contains(pnt):
                continue
            starting_locations.append([{
                'point': pnt,
                'edge_pos': None,
                'key_point': True
            }])

    return {
        'region': region.name,
        'search_rect': search_rect,
        'cache': cache,
        'starting_locations': {
            'junction': starting_locations,
            'middle': []
        },
        'gc': None
    }


def post_process_graph(graph_dict):
    save_dir = os.path.join(
        cfg.DIR.SAVE_GRAPH_DIR,
        '{}_{}'.format(cfg.TEST.CKPT, cfg.TEST.NUM_TARGETS),
        'post'
    )
    os.makedirs(save_dir, exist_ok=True)
    for region_name, g in graph_dict.items():
        bad_edges = set()
        road_segments, _ = graph_helper.get_graph_road_segments(g)
        for rs in road_segments:
            if rs.marked_length < 2 * cfg.TEST.STEP_LENGTH and \
                    (len(rs.src(g).in_edges_id) <= 1 or len(rs.dst(g).in_edges_id) <= 1):
                for edge in rs.edges(g):
                    bad_edges.add(edge)
        ng = graph_helper.Graph()
        seen_pnts = dict()
        for edge in g.edges.values():
            if edge in bad_edges:
                continue
            if edge.src(g).point == edge.dst(g).point:
                continue
            src_dst = []
            for pnt in [edge.src(g).point, edge.dst(g).point]:
                if pnt not in seen_pnts:
                    v = ng.add_vertex(pnt)
                    seen_pnts[pnt] = v.id
                src_dst.append(seen_pnts[pnt])
            ng.add_edge(src_dst[0], src_dst[1])
        ng.save(os.path.join(save_dir, '{}.graph'.format(
            region_name)), clear_self=False)


def prepare_net():
    print('initializing model')
    net = RPNet(cfg.TRAIN.NUM_TARGETS)
    net = net.cuda()
    file_name = os.path.join(cfg.DIR.CHECK_POINT_DIR,
                             '{}.pth.tar'.format(cfg.TEST.CKPT))
    if os.path.isfile(file_name):
        net = load_pretrained(net, file_name)
    if cfg.TEST.DATA_PARALLEL:
        net = torch.nn.DataParallel(net)
    return net


def generate_sample_lst(IMG_SZ, CROP_SZ, SAMPLE_STEP=2):
    CROP_SAMPLE_LST = []
    rows = list(range(0, IMG_SZ - CROP_SZ + 1, CROP_SZ // SAMPLE_STEP))
    cols = list(range(0, IMG_SZ - CROP_SZ + 1, CROP_SZ // SAMPLE_STEP))
    for r in rows:
        for c in cols:
            CROP_SAMPLE_LST.append((r, c))
    return CROP_SAMPLE_LST


def infer_segmentation(net, region_names):
    start_time = time.time()
    trans = transforms.ToTensor()
    CROP_SAMPLE_LST = generate_sample_lst(
        cfg.TEST.TEST_IMG_SZ, cfg.TEST.CROP_SZ)
    cuda_device_num = torch.cuda.device_count()
    road_map_dict = dict()
    junc_map_dict = dict()
    for num, region_name in enumerate(region_names):
        print("[{:2d}/{:2d}] {}".format(num, len(region_names), region_name))
        img_map = np.array(Image.open(os.path.join(
            cfg.DIR.TEST_IMAGERY_DIR, region_name + ".png")))
        img_map = img_map.swapaxes(0, 1)
        img_map = trans(img_map)
        img_map = torch.unsqueeze(img_map, 0)
        container = {}
        container['road'] = MapContainer(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road"),
                                         region_name, cfg.TEST.TEST_IMG_SZ)
        container['junc'] = MapContainer(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "junction"),
                                         region_name, cfg.TEST.TEST_IMG_SZ)
        pnt_index = 0
        pbar = tqdm(total=len(CROP_SAMPLE_LST))
        while pnt_index < len(CROP_SAMPLE_LST):
            pnt_lst = CROP_SAMPLE_LST[pnt_index:pnt_index +
                                      cfg.TEST.BATCH_SIZE_SEG]
            # bug: DataParallel, must feed something into every gpu
            if len(pnt_lst) < cuda_device_num:
                pnt_lst = CROP_SAMPLE_LST[:-cuda_device_num]
            batch_input = []
            for pnt in pnt_lst:
                crop_img = img_map[:, :, pnt[0]:pnt[0] +
                                   cfg.TEST.CROP_SZ, pnt[1]:pnt[1] + cfg.TEST.CROP_SZ]
                batch_input.append(crop_img)
            batch_input = torch.cat(batch_input, dim=0)
            input_var = torch.autograd.Variable(batch_input).cuda()
            res = net(input_var, None, test=True)
            road, junc = res['road'], res['junc']
            container['road'].add_batch_gpu(pnt_lst, road, cfg.TEST.CROP_SZ)
            container['junc'].add_batch_gpu(pnt_lst, junc, cfg.TEST.CROP_SZ)
            pnt_index += len(pnt_lst)
            pbar.update(len(pnt_lst))
        pbar.close()
        for item in container.values():
            item.close()
            item.save_map()
        road_map_dict[region_name] = container['road'].get_map().swapaxes(0, 1)
        junc_map_dict[region_name] = container['junc'].get_map().swapaxes(0, 1)
    duration = time.time() - start_time
    print('{} images, img_sz: {}, infer time: {}, speed: {}fps'.format(
        len(region_names), cfg.TEST.TEST_IMG_SZ, duration, len(region_names) / duration))
    return road_map_dict, junc_map_dict


def junction_nms(region_name, junc_map):
    print("  region: {}".format(region_name))
    junc_pnts = list()
    res_map = np.zeros(junc_map.shape)
    vis_map = np.zeros((junc_map.shape[0], junc_map.shape[1], 3))
    if np.max(junc_map) > 1:
        vis_map[:, :, 1] = junc_map
        junc_map[np.where(
            junc_map < cfg.TEST.BINARIZE_MAP.JUNC_SEG_THRESHOLE * 255)] = 0
    else:
        vis_map[:, :, 1] = junc_map * 255
        junc_map[np.where(
            junc_map < cfg.TEST.BINARIZE_MAP.JUNC_SEG_THRESHOLE)] = 0
    junc_map[np.where(junc_map)] = 1
    labels = measure.label(junc_map, connectivity=2)
    props = measure.regionprops(labels, coordinates='xy')
    for region in props:
        if region.area > cfg.TEST.BINARIZE_MAP.ANCHOR_MAX_REGION_AREA:
            continue
        center = (int(region.centroid[0]), int(region.centroid[1]))
        res_map[center] = 255
        cv.circle(vis_map, (center[1], center[0]),
                  radius=7, color=(0, 0, 255), thickness=-1)
        junc_pnts.append(center)
    cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                            "junc_nms", region_name + ".png"), res_map)
    cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT,
                            "junc_nms_vis", region_name + ".png"), vis_map)
    return junc_pnts


def road_seg_region_filter(region_name, road_seg):
    frame = road_seg.copy()
    frame[np.where(frame < cfg.TEST.BINARIZE_MAP.ROAD_SEG_THRESHOLE)] = 0
    frame[np.where(frame)] = 1
    frame = frame.astype(np.uint8)
    labels = measure.label(frame, connectivity=2)
    props = measure.regionprops(labels, coordinates='xy')
    for region in props.copy():
        if region.area < cfg.TEST.BINARIZE_MAP.MIN_BAD_ROAD_AREA:
            frame[tuple(region.coords.swapaxes(0, 1))] = 0
            props.remove(region)
    frame = cv.bitwise_and(road_seg, road_seg, mask=frame)
    cv.imwrite(os.path.join(cfg.DIR.SAVE_SEG_DIR, cfg.TEST.CKPT, "road_seg_region_filter", region_name + ".png"),
               frame*255)
    return frame


if __name__ == "__main__":
    with torch.no_grad():
        main()
