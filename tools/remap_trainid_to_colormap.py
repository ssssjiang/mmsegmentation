# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import numpy as np
from copy import deepcopy
from pathlib import Path
from mmcv import Config, DictAction

from mmseg.datasets import build_dataset

import torchsnooper
import snoop
from snoop import pp
from snoop.configuration import len_shape_watch, dtype_watch

torchsnooper.register_snoop(verbose=True)
snoop.install(color=True, columns=['time', 'function'],
              watch_extras=[len_shape_watch, dtype_watch])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--remap_mode', type=str, default='test')
    args = parser.parse_args()
    return args


def remap_to_colormap(dataset, cfg):
    palette = dataset.PALETTE
    if dataset.PALETTE is None:
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 255, size=(len(dataset.CLASSES), 3))
        np.random.set_state(state)

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))
    for idx in range(len(dataset.img_infos)):
        gt_segm = dataset.get_gt_seg_map_by_idx(idx)
        palette = np.array(palette)
        assert palette.shape[0] == len(dataset.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        color_seg = np.zeros((gt_segm.shape[0], gt_segm.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[gt_segm == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1].astype(np.uint8)

        color_seg_file = os.path.join(
            cfg['data_root'], cfg['ann_dir'].replace('ann', 'color_ann'),
            dataset.img_infos[idx]['filename'])
        pp(color_seg_file)

        mmcv.imwrite(color_seg, color_seg_file)
        prog_bar.update()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.remap_mode == 'test':
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
        cfg = cfg.data.test
    elif args.remap_mode == 'train':
        cfg = cfg.data.train
    elif args.remap_mode == 'val':
        cfg = cfg.data.val
    else:
        raise ValueError(f'Not support {args.stat_mode} mode.')

    dataset = build_dataset(cfg)
    remap_to_colormap(dataset, cfg)


if __name__ == '__main__':
    main()
