#!/bin/bash

# Single GPUs
python tools/test.py configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
   checkpoints/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth \
    --out work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/results.pkl \
    --show-dir /persist_datasets/cityscapes/results

# multiple GPUs test.
./tools/dist_test.sh configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
  checkpoints/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth \
   4 --out work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/results.pkl --format-only --eval cityscapes

python tools/confusion_matrix.py configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
  work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/results.pkl \
  work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes

./tools/dist_train.sh configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
  8 --work-dir work_dirs/bisenetv2_fcn_4x8_1024x1024_160k/ --deterministic

./tools/dist_train.sh  configs/ocrnet/ocrnet_hr18_512x1024_40k_nuimages.py  \
  4 --work-dir work_dirs/ocrnet_hr18_512x1024_40k_nuim/ --deterministic --no-validate

./tools/dist_train.sh  configs/ocrnet/ocrnet_hr18_512x1024_100_test_cityscapes2.py  \
  4 --work-dir work_dirs/ocrnet_hr18_512x1024_100_test_cityscapes2/ --deterministic