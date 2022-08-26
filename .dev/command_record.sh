#!/bin/bash

# Single GPU test.
python tools/test.py configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
   checkpoints/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth \
    --out work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/results.pkl \
    --show-dir /persist_datasets/cityscapes/results

python tools/test.py configs/ocrnet/ocrnet_hr18_512x1024_100_test_cityscapes1.py \
   work_dirs/ocrnet_hr18_512x1024_100_test_cityscapes4/iter_40000.pth \
    --out work_dirs/ocrnet_hr18_512x1024_100_test_cityscapes4/iter_40000/results.pkl \
    --show-dir work_dirs/ocrnet_hr18_512x1024_100_test_cityscapes4/iter_40000

python tools/test.py configs/ocrnet/ocrnet_hr18_512x1024_40k_nuimages.py \
   work_dirs/ocrnet_hr18_512x1024_100_test_cityscapes4/iter_40000.pth \
    --out work_dirs/ocrnet_hr18_512x1024_40k_nuimages/iter_40000/results.pkl \
    --show-dir work_dirs/ocrnet_hr18_512x1024_40k_nuimages/iter_40000

# # next!!
python tools/test.py configs/ocrnet/ocrnet_hr48_512x1024_20k_cityscapes_sidewalk.py \
  work_dirs/ocrnet_hr48_512x1024_40k_cityscapes_sidewalk/latest.pth \
  --out work_dirs/ocrnet_hr48_512x1024_40k_cityscapes_sidewalk/results.pkl \
  --show-dir work_dirs/ocrnet_hr48_512x1024_40k_cityscapes_sidewalk/show \
  --eval cityscapes --opacity 1 --gpu-id 6

# multiple GPUs test.
./tools/dist_test.sh configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
  checkpoints/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth \
   4 --out work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/results.pkl --format-only --eval cityscapes

./tools/dist_test.sh configs/ocrnet/ocrnet_hr48_512x1024_20k_cityscapes_sidewalk.py \
    work_dirs/ocrnet_hr48_512x1024_40k_cityscapes_sidewalk/latest.pth \
    4 --out work_dirs/ocrnet_hr48_512x1024_40k_cityscapes_sidewalk/results.pkl \
    --eval cityscapes

# confusion matrix
python tools/confusion_matrix.py configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
  work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes/results.pkl \
  work_dirs/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes

python tools/confusion_matrix.py configs/ocrnet/ocrnet_hr18_512x1024_100_test_cityscapes1.py \
  work_dirs/ocrnet_hr18_512x1024_100_test_cityscapes4/iter_40000/results.pkl \
  work_dirs/ocrnet_hr18_512x1024_100_test_cityscapes4/iter_40000

# multiple GPUs train.
./tools/dist_train.sh configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py \
  8 --work-dir work_dirs/bisenetv2_fcn_4x8_1024x1024_160k/ --deterministic

./tools/dist_train.sh  configs/ocrnet/ocrnet_hr18_512x1024_40k_nuimages.py  \
  4 --work-dir work_dirs/ocrnet_hr18_512x1024_40k_nuim/ --deterministic --no-validate

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh  configs/ocrnet/oocrnet_hr48_512x1024_20k_cityscapes_sidewalk.py  \
  4 --work-dir work_dirs/ocrnet_hr48_512x1024_40k_cityscapes_sidewalk/ --deterministic
