import argparse
import os
import time

import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict

import model.model as model
# from utils import crash_on_ipy
from utils.utils import AverageMeter, load_pretrained, numpy2tensor2cuda, get_logger
from utils.OSMDataset import OSMDataset
from torch.utils.tensorboard import SummaryWriter


def epoch_to_learning_rate(epoch):
    if epoch <= 20:
        return 1e-3
    else:
        return 1e-4


def main():
    parser = argparse.ArgumentParser(description="VecRoad Pytorch Train")
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

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.GPU_ID

    os.makedirs(cfg.DIR.DATA_ROOT, exist_ok=True)
    os.makedirs(cfg.DIR.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.DIR.CHECK_POINT_DIR, exist_ok=True)
    os.makedirs(cfg.DIR.SHORTCUT_DIR, exist_ok=True)

    logger = get_logger(logger_name="logtrain", log_dir=cfg.DIR.LOG_DIR)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.DIR.LOG_DIR))

    osm = OSMDataset(cfg)

    losses = AverageMeter()
    anchor_losses = AverageMeter()
    road_losses = AverageMeter()
    junc_losses = AverageMeter()
    time_meter = AverageMeter()

    net = model.RPNet(num_targets=cfg.TRAIN.NUM_TARGETS)

    if cfg.TRAIN.DATA_PARALLEL:
        net = torch.nn.DataParallel(net)
    net = net.cuda()

    criteria = lambda a, b: F.binary_cross_entropy_with_logits(a, b, reduction='sum')

    if cfg.TRAIN.SOLVER.METHOD == "Adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.TRAIN.SOLVER.LEARNING_RATE,
            betas=(0.9, 0.99), weight_decay=cfg.TRAIN.SOLVER.WEIGHT_DECAY)

    if cfg.TRAIN.LOAD_CHECK_POINT:
        file_name = os.path.join(cfg.DIR.CHECK_POINT_DIR, cfg.TRAIN.CHECK_POINT_NAME)
        if os.path.isfile(file_name):
            net, optimizer = load_pretrained(net, file_name, optimizer, strict=True)

    start_epoch = 1 if cfg.TRAIN.START_EPOCH == 0 else cfg.TRAIN.START_EPOCH

    for outer_it in range(start_epoch, cfg.TRAIN.TOTAL_ITERATION + 1):
        # adjust learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        expected_lr = epoch_to_learning_rate(outer_it)
        if current_lr != expected_lr:
            msg = "adjust learning rate: {}".format(expected_lr)
            logger.info(msg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = epoch_to_learning_rate(outer_it)
        else:
            msg = "current learning rate: {}".format(current_lr)
            logger.info(msg)

        # if outer_it > 10:
        #    FOLLOW_MODE = "follow_output"

        net.train()

        for path_it in range(2048):

            stage_time = time.time()

            data_dict = osm.get_batch()

            batch_inputs_cuda = numpy2tensor2cuda(data_dict.batch_inputs)
            batch_walked_path_cuda = numpy2tensor2cuda(data_dict.batch_walked_path_small)
            batch_target_maps_cuda = numpy2tensor2cuda(data_dict.batch_target_maps)
            batch_road_segmentation_cuda = numpy2tensor2cuda(data_dict.batch_road_segmentation)
            batch_junction_segmentation_cuda = numpy2tensor2cuda(data_dict.batch_junction_segmentation)

            """
            Net Processing
            """

            batch_output_cuda_dict = net(batch_inputs_cuda, batch_walked_path_cuda)

            batch_output_road_cuda = batch_output_cuda_dict['road']
            batch_output_junc_cuda = batch_output_cuda_dict['junc']
            batch_output_anchor_maps_cuda = batch_output_cuda_dict['anchor']
            batch_output_anchor_step_maps_cuda = batch_output_cuda_dict['anchor_middle']

            """
            Loss Calculation
            """

            anchor_loss = 0
            for i in range(cfg.TRAIN.BATCH_SIZE):
                inp = batch_output_anchor_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                target = batch_target_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                anchor_loss += criteria(inp, target).cuda()

            anchor_mid_loss = 0
            for i in range(cfg.TRAIN.BATCH_SIZE):
                inp = batch_output_anchor_step_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                target = batch_target_maps_cuda[i, :data_dict.batch_end_index[i], :, :]
                anchor_mid_loss += criteria(inp, target).cuda()

            summary_writer.add_scalar('anchor_loss', anchor_loss, outer_it * 2048 + path_it)
            summary_writer.add_scalar('anchor_mid_loss', anchor_mid_loss, outer_it * 2048 + path_it)

            anchor_loss += anchor_mid_loss

            # road_loss = junc_loss = 0
            # for item in batch_output_road_cuda:
            road_loss = criteria(batch_output_road_cuda, batch_road_segmentation_cuda).cuda()
            summary_writer.add_scalar('road_loss', road_loss, outer_it * 2048 + path_it)
            # for item in batch_output_junc_cuda:
            junc_loss = criteria(batch_output_junc_cuda, batch_junction_segmentation_cuda).cuda()
            summary_writer.add_scalar('junc_loss', junc_loss, outer_it * 2048 + path_it)


            loss = anchor_loss + road_loss + junc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1e4)
            optimizer.step()

            """
            Data post-processing
            """

            data_dict.batch_output_road = torch.sigmoid(batch_output_road_cuda).detach().cpu().numpy()
            data_dict.batch_output_junc = torch.sigmoid(batch_output_junc_cuda).detach().cpu().numpy()
            data_dict.batch_output_anchor_maps = torch.sigmoid(batch_output_anchor_maps_cuda).detach().cpu().numpy()

            losses.update(loss.data.item())
            anchor_losses.update(anchor_loss.data.item())
            road_losses.update(road_loss.data.item())
            junc_losses.update(junc_loss.data.item())

            time_meter.update(time.time() - stage_time)

            if path_it % cfg.TRAIN.PRINT_ITERATION == 0:
                msg = "iter:[{0}]-[{1}/2048] " \
                        "Time: {time_meter.val:.3f} ({time_meter.avg:.3f}) " \
                        "Anchor: {anchor_loss.val:.3f} ({anchor_loss.avg:.3f}) " \
                        "Road: {road_loss.val:.3f} ({road_loss.avg:.3f}) " \
                        "Junc: {junc_loss.val:.3f} ({junc_loss.avg:.3f}) " \
                        "Total: {total_loss.val:.3f} ({total_loss.avg:.3f})" \
                    .format(outer_it, path_it, time_meter=time_meter, anchor_loss=anchor_losses,
                            road_loss=road_losses, junc_loss=junc_losses, total_loss=losses)
                logger.info(msg)

            osm.push_and_vis_batch(data_dict, outer_it, path_it)

            if (path_it + 1) % cfg.TRAIN.SAVE_ITERATIONS == 0:
                msg = "iter:[{0}]-[{1}/2048] " \
                        "Time: {time_meter.sum:.3f} " \
                        "Anchor: {anchor_loss.avg:.3f} " \
                        "Road: {road_loss.avg:.3f} " \
                        "Junc: {junc_loss.avg:.3f} " \
                        "Total: {total_loss.avg:.3f}" \
                    .format(outer_it, path_it, time_meter=time_meter, anchor_loss=anchor_losses,
                            road_loss=road_losses, junc_loss=junc_losses, total_loss=losses)
                logger.info(msg)

                time_meter.reset()
                losses.reset()
                anchor_losses.reset()
                road_losses.reset()
                junc_losses.reset()
                if outer_it >= 30 or outer_it % 10 == 0:
                    save_file = os.path.join(cfg.DIR.CHECK_POINT_DIR, "{}.{}.pth.tar".format(outer_it, path_it))
                    torch.save({
                        "outer_it": outer_it,
                        "path_it": path_it,
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }, save_file)
    summary_writer.close()


if __name__ == '__main__':
    main()
