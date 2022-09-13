#!/usr/bin/env bash

CFG=configs/ocrnet/ocrnet_hr48_512x1024_40k_standard_sidewalk.py
WORK_DIR=work_dirs/ocrnet_hr48_512x1024_40k_standard_sidewalk/

# 'sunny' 'rainy' 'snowy' 'foggy'
WEATHER=('sunny' 'rainy' 'snowy' 'foggy')
# 'blur' 'night' 'shadow' 'overexposed' 'underexposed' 'screen' 'sun-flare'
CASE=('blur' 'night' 'shadow' 'overexposed' 'underexposed' 'screen' 'sun-flare')

# filter data
for weather in ${WEATHER[*]}
do
  echo "filter weather: ${weather}"
  python tools/filter_img_by_labels.py "${CFG}" \
    --filter_img_by_weather --weather "${weather}"
done

for corner_case in ${CASE[*]}
do
  echo "filter corner case: ${corner_case}"
  python tools/filter_img_by_labels.py "${CFG}" \
    --filter_img_by_case --case "${corner_case}"
done