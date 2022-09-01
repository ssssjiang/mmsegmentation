# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.cm
import matplotlib.pyplot as plt

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
    parser.add_argument('--plot_stats', action="store_true")
    parser.add_argument('--save_all_stats', action="store_true")
    parser.add_argument('--save_single_stats', action="store_true")
    parser.add_argument('--read_label_stats', action="store_true")
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--stat_mode', type=str, default='test')
    args = parser.parse_args()
    return args


def read_stat(dataset, cfg, idx):
    n = len(dataset.CLASSES)
    single_img_info_filename = \
        Path(dataset.img_infos[idx]['filename']).parent / \
        Path(dataset.img_infos[idx]['filename']).stem
    single_img_info_file = os.path.join(
        cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'),
        single_img_info_filename.with_suffix(".json"))
    assert os.path.exists(single_img_info_file)

    single_img_info = mmcv.load(single_img_info_file)
    single_stat = np.zeros(n)
    for idx, label in enumerate(dataset.CLASSES):
        single_stat[idx] = float(single_img_info[label])
    return single_stat


def stat_labels(dataset, cfg, args):
    n = len(dataset.CLASSES)
    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))
    # ignore_index = dataset.ignore_index
    # pp(dataset.img_infos)

    all_labels_stat = np.zeros(n, dtype=np.float64)
    single_label_stat = []
    for idx in range(len(dataset.img_infos)):
        if not args.overwrite:
            single_img_info_filename = \
                Path(dataset.img_infos[idx]['filename']).parent / \
                Path(dataset.img_infos[idx]['filename']).stem
            single_img_info_file = os.path.join(
                cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'),
                single_img_info_filename.with_suffix(".json"))

            if os.path.exists(single_img_info_file):
                single_img_info = mmcv.load(single_img_info_file)
                single_stat = np.zeros(n)
                for idx, label in enumerate(dataset.CLASSES):
                    single_stat[idx] = float(single_img_info[label])

                single_label_stat.append(single_stat)
                all_labels_stat += single_stat

                prog_bar.update()
                continue

        gt_segm = dataset.get_gt_seg_map_by_idx(idx)
        gt_segm = gt_segm.flatten()
        # to_ignore = gt_segm == ignore_index
        # gt_segm = gt_segm[~to_ignore]
        single_stat = np.bincount(gt_segm, minlength=n)[: n]
        single_stat = single_stat / (single_stat.sum() + 0.0001)

        single_label_stat.append(single_stat)
        # all_labels_stat += single_stat

        del gt_segm
        prog_bar.update()

    all_labels_stat = np.mean(np.array(single_label_stat), axis=1)
    pp(all_labels_stat)

    # single_label_stat = np.array(single_label_stat)
    # pp(single_label_stat)
    return all_labels_stat, single_label_stat


def save_all_imgs_info(dataset, all_labels_stat, cfg):
    all_imgs_info = deepcopy(cfg)
    all_imgs_info.pop('pipeline', None)
    all_imgs_info.pop('test_mode', None)

    for idx, label in enumerate(dataset.CLASSES):
        all_imgs_info[label] = float(all_labels_stat[idx])

    all_imgs_info_filename = Path(all_imgs_info['split']).stem \
        if 'split' in all_imgs_info.keys() else \
        Path(all_imgs_info['img_dir']).stem

    all_imgs_info_file = os.path.join(
        all_imgs_info['data_root'],
        all_imgs_info['ann_dir'].replace('ann', 'stat_ann'),
        all_imgs_info_filename + ".json")
    mmcv.dump(all_imgs_info, all_imgs_info_file, indent=4)


def save_single_imgs_info(dataset, single_label_stat, cfg, args):
    prog_bar = mmcv.ProgressBar(len(single_label_stat))

    for idx, label_stat in enumerate(single_label_stat):
        single_img_info_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem
        single_img_info_file = os.path.join(
            cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'),
            single_img_info_filename.with_suffix(".json"))

        if not args.overwrite and os.path.exists(single_img_info_file):
            prog_bar.update()
            continue

        single_img_info = dict()
        single_img_info['classes'] = cfg['classes']

        for label_id, label in enumerate(dataset.CLASSES):
            single_img_info[label] = float(label_stat[label_id])

        mmcv.dump(single_img_info, single_img_info_file, indent=4)
        prog_bar.update()


def plot_all_labels_stat(dataset, all_labels_stat, cfg):
    plt.bar(dataset.CLASSES, all_labels_stat, width=0.55, color='#87CEFA')

    for i in range(len(all_labels_stat)):
        plt.annotate(str(round(all_labels_stat[i], 3)),
                     xy=(dataset.CLASSES[i], all_labels_stat[i]),
                     ha='center', va='bottom')

    plt.legend()
    plt.ylabel('label pixel count')
    plt.xlabel('classes')

    all_labels_stat_filename = Path(cfg['split']).stem \
        if 'split' in cfg.keys() else \
        Path(cfg['img_dir']).stem
    all_labels_stat_file = os.path.join(
        cfg['data_root'],
        cfg['ann_dir'].replace('ann', 'stat_ann'),
        all_labels_stat_filename + ".png")
    plt.title(all_labels_stat_filename)
    plt.savefig(all_labels_stat_file, bbox_inches='tight')


def read_stats(dataset, cfg):
    single_img_info_path = os.path.join(
        cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'))
    assert os.path.exists(single_img_info_path)
    pp(single_img_info_path)

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))

    n = len(dataset.CLASSES)
    all_labels_stat = np.zeros(n, dtype=np.float64)
    single_label_stat = []
    for idx in range(len(dataset.img_infos)):
        single_stat = read_stat(dataset, cfg, idx)
        single_label_stat.append(single_stat)
        all_labels_stat += single_stat
        prog_bar.update()

    all_labels_stat = all_labels_stat / len(dataset.img_infos)
    pp(all_labels_stat)
    return all_labels_stat, single_label_stat


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.stat_mode == 'test':
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
        cfg = cfg.data.test
    elif args.stat_mode == 'train':
        cfg = cfg.data.train
    elif args.stat_mode == 'val':
        cfg = cfg.data.val
    else:
        raise ValueError(f'Not support {args.stat_mode} mode.')

    dataset = build_dataset(cfg)
    if args.read_label_stats:
        all_labels_stat, single_label_stat = read_stats(dataset, cfg)
    else:
        all_labels_stat, single_label_stat = stat_labels(dataset, cfg, args)

    if args.save_all_stats:
        save_all_imgs_info(dataset, all_labels_stat, cfg)

    if args.save_single_stats:
        save_single_imgs_info(dataset, single_label_stat, cfg, args)

    if args.plot_stats:
        plot_all_labels_stat(dataset, all_labels_stat, cfg)


if __name__ == '__main__':
    main()
