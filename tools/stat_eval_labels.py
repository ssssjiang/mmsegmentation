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

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'seg_show_dir', help='prediction path')
    parser.add_argument('--overwrite', type=bool, default=False)
    args = parser.parse_args()
    return args


def stat_error_labels(dataset, cfg, args):
    n = len(dataset.CLASSES)

    ignore_index = dataset.ignore_index

    err_stat_by_all = []
    err_stat_by_class = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for idx in range(len(dataset)):
        single_img_info_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem

        if not args.overwrite:
            err_stat_info_file = \
                Path(args.seg_show_dir).parent / 'err_stat' / \
                single_img_info_filename.with_suffix(".json")

            if os.path.exists(err_stat_info_file):
                err_stat_info = mmcv.load(err_stat_info_file)

                err_info_by_class = err_stat_info['err_base_class']
                err_info_by_all = err_stat_info['err_base_all']

                single_stat = np.zeros(n)
                single_class_stat = np.zeros(n)
                for idx, label in enumerate(dataset.CLASSES):
                    single_stat[idx] = float(err_info_by_all[label])
                    single_class_stat[idx] = float(err_info_by_class[label])

                err_stat_by_class.append(single_class_stat)
                err_stat_by_all.append(single_stat)

                prog_bar.update()
                continue

        single_img_info_file = os.path.join(
            cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'),
            single_img_info_filename.with_suffix(".json"))

        assert os.path.exists(single_img_info_file)

        single_img_info = mmcv.load(single_img_info_file)
        class_stat = np.zeros(n)
        for cls_id, label in enumerate(dataset.CLASSES):
            class_stat[cls_id] = float(single_img_info[label])

        res_segm_file = Path(args.seg_show_dir) / dataset.img_infos[idx]['filename']
        img_bytes = mmcv.FileClient(**dict(backend='disk')).get(res_segm_file)
        res_segm = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend='pillow').squeeze().astype(np.uint8)

        gt_segm = dataset.get_gt_seg_map_by_idx(idx).astype(int)
        gt_segm, res_segm = gt_segm.flatten(), res_segm.flatten()
        to_ignore = gt_segm == ignore_index
        gt_segm, res_segm = gt_segm[~to_ignore], res_segm[~to_ignore]
        inds = n * gt_segm + res_segm
        mat = np.bincount(inds, minlength=n ** 2).reshape(n, n)
        per_label_sums = mat.sum(axis=1)[:, np.newaxis]
        mat = mat.astype(np.float32) / (per_label_sums + 0.0001)

        single_class_stat = np.zeros(n)
        for label_id in range(n):
            if class_stat[label_id] == 0:
                single_class_stat[label_id] = 0
                continue
            single_class_stat[label_id] = 1 - mat[label_id][label_id]
        err_stat_by_class.append(single_class_stat)

        single_stat = single_class_stat * class_stat
        err_stat_by_all.append(single_stat)

        if idx % 100 == 0:
            pp(single_stat)
            pp(single_class_stat)

        del res_segm, gt_segm
        prog_bar.update()

    return err_stat_by_class, err_stat_by_all


def save_err_stat_info(dataset, err_stat_by_class, err_stat_by_all, cfg, args):
    prog_bar = mmcv.ProgressBar(len(err_stat_by_all))

    for idx in range(len(err_stat_by_all)):
        err_stat_info_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem
        err_stat_info_file = os.path.join(
            Path(args.seg_show_dir).parent / 'err_stat' / \
            err_stat_info_filename.with_suffix(".json"))

        if not args.overwrite and os.path.exists(err_stat_info_file):
            prog_bar.update()
            continue

        err_stat_info = dict()
        err_stat_info['classes'] = cfg['classes']

        err_info_by_class = dict()
        err_info_by_all = dict()
        for label_id, label in enumerate(dataset.CLASSES):
            err_info_by_class[label] = float(err_stat_by_class[idx][label_id])
            err_info_by_all[label] = float(err_stat_by_all[idx][label_id])

        err_stat_info['err_base_class'] = err_info_by_class
        err_stat_info['err_base_all'] = err_info_by_all

        mmcv.dump(err_stat_info, err_stat_info_file, indent=4)
        prog_bar.update()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.test)
    err_stat_by_class, err_stat_by_all = stat_error_labels(dataset, cfg.data.test, args)
    save_err_stat_info(dataset, err_stat_by_class, err_stat_by_all, cfg.data.test, args)


if __name__ == '__main__':
    main()
