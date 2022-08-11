import os.path as osp

import mmcv
from mmcv.utils import print_log

from .builder import DATASETS
from .custom import CustomDataset
from ..utils import get_root_logger


@DATASETS.register_module()
class NuImagesDataset(CustomDataset):
    '''CustomDatasets'''

    CLASSES = (
        'animal',
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer',
        'human.pedestrian.stroller',
        'human.pedestrian.wheelchair',
        'movable_object.barrier',
        'movable_object.debris',
        'movable_object.pushable_pullable',
        'movable_object.trafficcone',
        'static_object.bicycle_rack',
        'vehicle.bicycle',
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.car',
        'vehicle.construction',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
        'vehicle.motorcycle',
        'vehicle.trailer',
        'vehicle.truck',
        'flat.driveable_surface',
        'vehicle.ego',)

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200]]

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        # Here we take ann_dir as the annotation path
        annotations = mmcv.load(split)
        img_infos = []
        for img in annotations['images']:
            img_info = dict(filename=img['file_name'])
            seg_map = img_info['filename'].replace(img_suffix, seg_map_suffix)
            img_info['ann'] = dict(seg_map=osp.join('semantic_masks', seg_map))
            img_infos.append(img_info)

        print_log(
            f'Loaded {len(img_infos)} images from {ann_dir}',
            logger=get_root_logger())
        return img_infos
