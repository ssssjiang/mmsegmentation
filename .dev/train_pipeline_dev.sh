#!/usr/bin/env bash

CFG=configs/ocrnet/ocrnet_hr48_512x1024_40k_standard_sidewalk.py
WORK_DIR=work_dirs/ocrnet_hr48_512x1024_40k_standard_sidewalk/

# Train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./tools/dist_train.sh "${CFG}"  \
  6 --work-dir "${WORK_DIR}" --deterministic

mkdir ${WORK_DIR}/latest_eval
LATEST_EVAL=${WORK_DIR}/latest_eval

## Single GPU Test
#CUDA_VISIBLE_DEVICES=5 tools/dist_test.sh "${CFG}" \
#    "${WORK_DIR}"/latest.pth \
#    1  --show-seg-dir "${LATEST_EVAL}"/show_seg

# Test
CUDA_VISIBLE_DEVICES=5 tools/dist_test.sh"${CFG}" \
  "${WORK_DIR}"/latest.pth \
  1 --out "${LATEST_EVAL}"/simple-results.pkl \
  --show-dir "${LATEST_EVAL}"/show \
  --show-seg-dir "${LATEST_EVAL}"/show_seg \
  --eval mIoU --opacity 1  > "${LATEST_EVAL}"/latest_eval.log

# confusion matrix
python tools/confusion_matrix.py "${CFG}" \
  "${LATEST_EVAL}"/show_seg \
  "${LATEST_EVAL}"


# filter eval results
python tools/filter_img_by_labels.py "${CFG}" \
  "${LATEST_EVAL}"/err_stat \
  --filter_eval