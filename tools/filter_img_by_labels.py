# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import json

import mmcv
import numpy as np
from pathlib import Path
import random
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
    parser.add_argument('--label', type=str, default='sidewalk')
    parser.add_argument('--weather', type=str, default='sunny')
    parser.add_argument('--case', type=str, default='blur')
    parser.add_argument('--stat_mode', type=str, default='test')
    parser.add_argument('--eval_dir', type=str, default='',
                        help='prediction path where test err_stat result')
    parser.add_argument('--filter_img_by_weather', action='store_true')
    parser.add_argument('--filter_img_by_case', action='store_true')
    parser.add_argument('--filter_img_by_class', action='store_true')
    parser.add_argument('--filter_eval', action='store_true')
    args = parser.parse_args()
    return args


def read_eval_stats(dataset, cfg, args):
    n = len(dataset.CLASSES)
    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))

    err_stat_by_all = []
    err_stat_by_class = []
    for idx in range(len(dataset.img_infos)):
        single_img_info_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem
        err_stat_info_file = Path(args.eval_dir) / \
                             single_img_info_filename.with_suffix(".json")

        assert os.path.exists(err_stat_info_file)
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
    return np.array(err_stat_by_class), np.array(err_stat_by_all)


def filter_eval_by_err(
        dataset, err_stat_by_class, err_stat_by_all, cfg, args):
    label_id = dataset.CLASSES.index(args.label)
    mean_err_by_class = np.mean(err_stat_by_class, axis=0)[label_id]
    pp(np.mean(err_stat_by_class, axis=0))
    mean_err_by_all = np.mean(err_stat_by_all, axis=0)[label_id]
    pp(np.mean(err_stat_by_all, axis=0))

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))
    filtered_evals = []
    for idx in range(len(err_stat_by_class)):
        filtered_eval_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem
        if err_stat_by_class[idx][label_id] >= mean_err_by_class:
            filtered_evals.append(filtered_eval_filename.as_posix())
        prog_bar.update()

    pp(filtered_evals)
    pp(len(filtered_evals))

    with open(Path(args.eval_dir) / f'filtered_evals_{args.label}.txt', 'w') as f:
        for i in filtered_evals:
            f.write(i + '\n')


def read_stats(dataset, single_img_info_path, idx):
    single_img_info_filename = \
        Path(dataset.img_infos[idx]['filename']).parent / \
        Path(dataset.img_infos[idx]['filename']).stem
    single_img_info_file = os.path.join(
        single_img_info_path,
        single_img_info_filename.with_suffix(".json"))
    assert os.path.exists(single_img_info_file)

    single_img_info = mmcv.load(single_img_info_file)
    return single_img_info


def read_label_stats(dataset, cfg):
    single_img_info_path = os.path.join(
        cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'))
    assert os.path.exists(single_img_info_path)
    pp(single_img_info_path)

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))

    n = len(dataset.CLASSES)
    all_labels_stat = np.zeros(n, dtype=np.float64)
    single_label_stat = []
    for idx in range(len(dataset.img_infos)):
        single_img_info = read_stats(dataset, single_img_info_path, idx)
        single_stat = np.zeros(n)
        for idx, label in enumerate(dataset.CLASSES):
            single_stat[idx] = float(single_img_info[label])

        single_label_stat.append(single_stat)
        all_labels_stat += single_stat
        prog_bar.update()

    all_labels_stat = all_labels_stat / len(dataset.img_infos)
    pp(all_labels_stat)
    return all_labels_stat, single_label_stat


def update_img_stat(dataset, args, cfg):
    single_img_info_path = os.path.join(
        cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'))
    assert os.path.exists(single_img_info_path)
    pp(single_img_info_path)

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))

    updated_imgs = []
    for idx in range(len(dataset.img_infos)):
        single_img_info_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem
        single_img_info_file = os.path.join(
            single_img_info_path,
            single_img_info_filename.with_suffix(".json"))
        assert os.path.exists(single_img_info_file)

        single_img_info = mmcv.load(single_img_info_file)

        # TODO: add update context.
        # if "classes" in single_img_info.keys():
        #     classes = single_img_info.pop("classes")
        #     single_img_info["classes"] = dict()
        #     for label in classes:
        #         single_img_info["classes"].update({label: single_img_info[label]})

        mmcv.dump(single_img_info, single_img_info_file, indent=4)
        updated_img_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem
        updated_imgs.append(updated_img_filename)

        prog_bar.update()
    pp(updated_imgs)


def filter_img_by_weather(dataset, args, cfg):
    single_img_info_path = os.path.join(
        cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'))
    assert os.path.exists(single_img_info_path)
    pp(single_img_info_path)

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))

    filtered_imgs = []
    for idx in range(len(dataset.img_infos)):
        single_img_info = read_stats(dataset, single_img_info_path, idx)
        filtered_img_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem

        selected = False
        if args.filter_img_by_weather and \
                "weather" in single_img_info.keys() and \
                single_img_info["weather"] == args.weather:
            selected = True

        if args.filter_img_by_case and \
                "corner-case" in single_img_info.keys() and \
                args.case in single_img_info["corner-case"].keys():
            selected = True

        if selected:
            filtered_imgs.append(filtered_img_filename)
        prog_bar.update()

    pp(len(filtered_imgs))
    filtered_imgs_dir = Path(cfg['data_root']) / Path(cfg['img_dir'])
    tag = ''
    if args.filter_img_by_weather:
        tag += args.weather
    if args.filter_img_by_case:
        tag += args.case
    with open(filtered_imgs_dir / f'filtered_imgs_{tag}.txt', 'w') as f:
        for i in filtered_imgs:
            f.write(str(i) + '\n')


def filter_normer_imgs(dataset, args, cfg):
    single_img_info_path = os.path.join(
        cfg['data_root'], cfg['ann_dir'].replace('ann', 'stat_ann'))
    assert os.path.exists(single_img_info_path)
    pp(single_img_info_path)

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))

    filtered_imgs = []
    for idx in range(len(dataset.img_infos)):
        single_img_info = read_stats(dataset, single_img_info_path, idx)
        filtered_img_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem

        if "weather" in single_img_info.keys() and \
                single_img_info["weather"] == "sunny" and \
                "corner-case" not in single_img_info.keys():
            filtered_imgs.append(filtered_img_filename)
        prog_bar.update()

    pp(len(filtered_imgs))
    filtered_imgs_dir = Path(cfg['data_root']) / Path(cfg['img_dir'])
    with open(filtered_imgs_dir / f'filtered_imgs_normal.txt', 'w') as f:
        for i in filtered_imgs:
            f.write(str(i) + '\n')


def filter_img_by_class(
        dataset, all_labels_stat, single_label_stat, args, cfg):
    label_id = dataset.CLASSES.index(args.label)

    prog_bar = mmcv.ProgressBar(len(dataset.img_infos))
    filtered_imgs = []
    sample_imgs = []
    for idx, label_stat in enumerate(single_label_stat):
        filtered_img_filename = \
            Path(dataset.img_infos[idx]['filename']).parent / \
            Path(dataset.img_infos[idx]['filename']).stem
        if label_stat[label_id] >= all_labels_stat[label_id]:
            filtered_imgs.append(filtered_img_filename.as_posix())
        else:
            sample_imgs.append(filtered_img_filename.as_posix())
        prog_bar.update()

    sample_imgs = random.sample(sample_imgs, 2 * len(sample_imgs) // 3)
    pp(len(filtered_imgs))
    pp(len(sample_imgs))
    filtered_imgs_dir = Path(cfg['data_root']) / Path(cfg['img_dir'])
    with open(filtered_imgs_dir / f'filtered_imgs_{args.label}.txt', 'w') as f:
        for i in filtered_imgs * 2 + sample_imgs:
            f.write(i + '\n')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.filter_eval:
        assert args.stat_mode in ['test', 'val']

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
    assert args.label in dataset.CLASSES

    if args.filter_img_by_class:
        all_labels_stat, single_label_stat = read_stats(dataset, cfg)
        filter_img_by_class(
            dataset, all_labels_stat, single_label_stat, args, cfg)
    if args.filter_img_by_weather or args.filter_img_by_case:
        filter_img_by_weather(dataset, args, cfg)
        # update_img_stat(dataset, args, cfg)

    # filter_normer_imgs(dataset, args, cfg)

    if args.filter_eval:
        assert os.path.exists(args.eval_dir)

        err_stat_by_class, err_stat_by_all = \
            read_eval_stats(dataset, cfg, args)
        filter_eval_by_err(
            dataset, err_stat_by_class, err_stat_by_all, cfg, args)


if __name__ == '__main__':
    main()
