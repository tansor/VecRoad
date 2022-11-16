import os
import random
import numpy as np
import time
import logging
from tqdm import tqdm

import utils.model_utils as model_utils
from utils.tileloader import Tiles
from easydict import EasyDict


class OSMDataset:

    def __init__(self, cfg, training=True, seg_input=None):
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.window_size = cfg.TRAIN.WINDOW_SIZE
        self.input_channels = cfg.TRAIN.NUM_INPUT_CHANNELS
        self.seg_input = seg_input
        self.num_targets = cfg.TRAIN.NUM_TARGETS
        self.paths = []
        self.tiles = Tiles(training_regions=self.cfg.TRAIN.TRAINING_REGIONS, 
                           parallel_tiles=self.cfg.TRAIN.PARALLEL_TILES, 
                           region_path=cfg.DIR.IMG_COORD_PATH,
                           graph_dir=cfg.DIR.GRAPH_DIR, 
                           tile_dir=cfg.DIR.TILE_DIR)
        self.save_idx = 0
        self.training = training
        self.subtiles = self.tiles.prepare_training()
        print("extracted {} subtiles from {} tiles (missing {})".format(
            len(self.subtiles), len(self.tiles.train_tiles), 4 * len(self.tiles.train_tiles) - len(self.subtiles)))
        print("loading initial paths")
        self.paths = []
        for i, subtile in enumerate(self.subtiles):
            self.paths.append(model_utils.Path(i, training, subtile["gc"].clone(), subtile))

    def warm_up(self):
        print("warm up now:")
        for path_idx in tqdm(range(len(self.paths))):
            path = self.paths[path_idx]
            for i in range(random.randint(self.cfg.TRAIN.MAX_PATH_LENGTH//4, self.cfg.TRAIN.MAX_PATH_LENGTH)):
                while True:
                    extension_vertex, is_key_point = path.pop(follow_order=False, probs=[0.2, 0.8, 0],
                                                              WINDOW_SIZE=self.window_size)
                    if extension_vertex is None or len(path.graph.vertices) >= self.cfg.TRAIN.MAX_PATH_LENGTH:
                        self.paths[path_idx] = model_utils.Path(
                            idx=path_idx, training=self.training, gc=self.subtiles[path_idx]["gc"].clone(),
                            tile_data=self.subtiles[path_idx])
                        path = self.paths[path_idx]
                        continue
                    break
                target_poses = path.get_target_poses(
                    extension_vertex=extension_vertex, road_segmentation=None,
                    STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH, is_key_point=is_key_point,
                    NUM_TARGETS=self.num_targets, RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS,
                    WINDOW_SIZE=self.window_size)
                if extension_vertex.edge_pos is None:
                    continue
                if len(target_poses) == 0:
                    continue
                if is_key_point:
                    length = len(target_poses.target_poses[0])
                    if length > 0:
                        target_poses.target_poses[0] = \
                            random.sample(target_poses.target_poses[0], random.randint(1, length))
                path.push(
                    extension_vertex=extension_vertex, is_key_point=is_key_point,
                    follow_mode=self.cfg.TRAIN.FOLLOW_MODE, target_poses=target_poses,
                    output_points=None,
                    RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS,
                    road_segmentation=None,
                    NUM_TARGETS=self.cfg.TRAIN.NUM_TARGETS, WINDOW_SIZE=self.cfg.TRAIN.WINDOW_SIZE,
                    STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH,
                    AVG_CONFIDENCE_THRESHOLD=self.cfg.TRAIN.AVG_CONFIDENCE_THRESHOLD)

    def get_batch(self):
        path_indices = random.sample(range(len(self.paths)), self.batch_size)
        batch_extension_vertices = []
        batch_inputs = np.zeros((self.batch_size, self.input_channels, self.window_size, self.window_size))
        batch_target_maps = np.zeros((self.batch_size, self.num_targets, self.window_size, self.window_size))
        batch_is_key_point = np.zeros(self.batch_size)
        batch_end_index = np.zeros(self.batch_size, dtype=np.int)
        batch_target_poses = []
        default_shape = (self.batch_size, 1, self.window_size, self.window_size)
        batch_walked_path_small = np.zeros((self.batch_size, 1, self.window_size // 4, self.window_size // 4))
        batch_road_segmentation = np.zeros((self.batch_size, 1, self.window_size // 4, self.window_size // 4))
        batch_road_segmentation_thick3 = np.zeros(default_shape)
        batch_junction_segmentation = np.zeros((self.batch_size, 1, self.window_size // 4, self.window_size // 4))
        batch_aerial_images_hwc = []

        for i in range(len(path_indices)):
            path_idx = path_indices[i]
            path = self.paths[path_idx]

            while True:
                extension_vertex, is_key_point = path.pop(follow_order=False, probs=[0.15, 0.8, 0.05],
                                                          WINDOW_SIZE=self.window_size)

                if extension_vertex is None or len(path.graph.vertices) >= self.cfg.TRAIN.MAX_PATH_LENGTH:
                    self.paths[path_idx] = model_utils.Path(
                        idx=path_idx, training=self.training, gc=self.subtiles[path_idx]["gc"].clone(),
                        tile_data=self.subtiles[path_idx])
                    path = self.paths[path_idx]
                    continue
                break

            fetch_list = ['aerial_image_chw',
                          'aerial_image_hwc',
                          'walked_path_small',
                          'road_seg_small',
                          'road_seg_thick3',
                          'junc_seg_small']

            data_dict = path.make_path_input(extension_vertex=extension_vertex,
                                             fetch_list=fetch_list,
                                             is_key_point=is_key_point,
                                             WINDOW_SIZE=self.window_size)
            data_dict = EasyDict(data_dict)

            target_poses = self.paths[path_idx].get_target_poses(
                extension_vertex=extension_vertex, road_segmentation=data_dict.road_seg_thick3[0],
                STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH, is_key_point=is_key_point,
                NUM_TARGETS=self.num_targets, RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS,
                WINDOW_SIZE=self.window_size)  # edge_pos list

            batch_aerial_images_hwc.append(data_dict.aerial_image_hwc)
            batch_extension_vertices.append(extension_vertex)
            batch_inputs[i] = data_dict.aerial_image_chw
            batch_walked_path_small[i] = data_dict.walked_path_small
            batch_road_segmentation[i] = data_dict.road_seg_small
            batch_road_segmentation_thick3[i] = data_dict.road_seg_thick3
            batch_junction_segmentation[i] = data_dict.junc_seg_small
            batch_target_poses.append(target_poses)
            batch_is_key_point[i] = is_key_point
            batch_end_index[i] = 1 if is_key_point else target_poses.get_supervision_end_index()

            target_maps = path.generate_target_maps(extension_vertex, target_poses, self.num_targets,
                                                    self.window_size,
                                                    is_key_point)
            batch_target_maps[i] = target_maps

        data = EasyDict({
            'path_indices': path_indices,
            'batch_extension_vertices': batch_extension_vertices,
            'batch_inputs': batch_inputs,
            'batch_target_maps': batch_target_maps,
            'batch_is_key_point': batch_is_key_point,
            'batch_end_index': batch_end_index,
            'batch_target_poses': batch_target_poses,
            'batch_walked_path_small': batch_walked_path_small,
            'batch_road_segmentation': batch_road_segmentation,
            'batch_road_segmentation_thick3': batch_road_segmentation_thick3,
            'batch_junction_segmentation': batch_junction_segmentation,
            'batch_aerial_images_hwc': batch_aerial_images_hwc
        })
        return data

    def push_and_vis_batch(self, res_dict, outer_it, path_it):

        if self.cfg.TRAIN.FOLLOW_MODE == "follow_output":
            batch_output_points = \
                model_utils.map_to_coordinate(
                    batch_output_maps=res_dict.batch_output_anchor_maps.copy(),
                    batch_is_key_point=res_dict.batch_is_key_point,
                    batch_extension_vertices=res_dict.batch_extension_vertices,
                    SEGMENTATION_THRESHOLD=self.cfg.TRAIN.BINARIZE_MAP.SEGMENTATION_THRESHOLD,
                    STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH,
                    MAX_REGION_AREA=self.cfg.TRAIN.BINARIZE_MAP.MAX_REGION_AREA)

        if self.cfg.TRAIN.SAVE_EXAMPLES and self.save_idx in res_dict.path_indices:
            x = res_dict.path_indices.index(self.save_idx)
            fname = os.path.join(self.cfg.DIR.SHORTCUT_DIR,
                                 "{}_{}_{}_".format(res_dict.path_indices[x], outer_it, path_it))

            self.paths[res_dict.path_indices[x]].visualize_output(
                fname_prefix=fname,
                extension_vertex=res_dict.batch_extension_vertices[x],
                aerial_image=res_dict.batch_aerial_images_hwc[x], target_poses=res_dict.batch_target_poses[x],
                pred_gt_pair_list=[
                    ("anchor", res_dict.batch_output_anchor_maps[x], res_dict.batch_target_maps[x]),
                    ("road", res_dict.batch_output_road[x, 0], res_dict.batch_road_segmentation[x, 0]),
                    ("junc", res_dict.batch_output_junc[x, 0], res_dict.batch_junction_segmentation[x, 0])
                ])

        for i in range(len(res_dict.path_indices)):
            if res_dict.batch_extension_vertices[i].edge_pos is None:
                continue
            if len(res_dict.batch_target_poses[i]) == 0:
                continue
            path_idx = res_dict.path_indices[i]
            if res_dict.batch_is_key_point[i]:
                if self.cfg.TRAIN.FOLLOW_MODE == "follow_target":
                    length = len(res_dict.batch_target_poses[i].target_poses[0])
                    if length > 0:
                        res_dict.batch_target_poses[i].target_poses[0] = \
                            random.sample(res_dict.batch_target_poses[i].target_poses[0], random.randint(1, length))
                elif self.cfg.TRAIN.FOLLOW_MODE == "follow_output":
                    length = len(batch_output_points[i])
                    if length > 0:
                        batch_output_points[i] = \
                            random.sample(batch_output_points[i], random.randint(1, length))
            self.paths[path_idx].push(
                extension_vertex=res_dict.batch_extension_vertices[i], is_key_point=res_dict.batch_is_key_point[i],
                follow_mode=self.cfg.TRAIN.FOLLOW_MODE, target_poses=res_dict.batch_target_poses[i],
                output_points=batch_output_points[i] if self.cfg.TRAIN.FOLLOW_MODE == "follow_output" else None,
                RECT_RADIUS=self.cfg.TRAIN.RECT_RADIUS, road_segmentation=res_dict.batch_road_segmentation_thick3[i, 0],
                NUM_TARGETS=self.cfg.TRAIN.NUM_TARGETS, WINDOW_SIZE=self.cfg.TRAIN.WINDOW_SIZE,
                STEP_LENGTH=self.cfg.TRAIN.STEP_LENGTH,
                AVG_CONFIDENCE_THRESHOLD=self.cfg.TRAIN.AVG_CONFIDENCE_THRESHOLD)
        return
