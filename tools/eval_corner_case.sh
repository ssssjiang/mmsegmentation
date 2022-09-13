#!/usr/bin/env bash

CFG=configs/ocrnet/ocrnet_hr48_512x1024_40k_standard_sidewalk.py
WORK_DIR=work_dirs/ocrnet_hr48_512x1024_40k_standard_sidewalk/

# 'sunny' 'rainy' 'snowy' 'foggy'
WEATHER=('sunny' 'rainy' 'snowy' 'foggy')
# 'blur' 'night' 'shadow' 'overexposed' 'underexposed' 'screen' 'sun-flare'
CASE=('blur' 'night' 'shadow' 'overexposed' 'underexposed' 'screen' 'sun-flare')

corner_case='night'
# eval data
# not support loop
echo "eval corner case: ${corner_case}"
EVAL_DIR="${WORK_DIR}/${corner_case}"
mkdir "${EVAL_DIR}"
mkdir "${EVAL_DIR}/show"
mkdir "${EVAL_DIR}/show_seg"

CUDA_VISIBLE_DEVICES=5 python tools/test.py "${CFG}" \
  "${WORK_DIR}"/latest.pth  \
  --out "${EVAL_DIR}"/simple-results.pkl \
  --show-dir "${EVAL_DIR}"/show \
  --show-seg-dir "${EVAL_DIR}"/show_seg \
  --eval mIoU --opacity 1  > "${EVAL_DIR}"/latest_eval.log
