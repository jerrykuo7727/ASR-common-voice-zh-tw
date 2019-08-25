#!/bin/bash

python3_cmd=python3
export DATA_DIR=/home/jiazhi/Dataset/common-voice_zh-TW_43h_2019-06-12

. ./path.sh || exit 1
. ./cmd.sh || exit 1

nj=4
lm_order=1

. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }


# Removing previously created data (from last run.sh execution)
rm -rf data mfcc exp  

echo
echo "===== PREPARING REQUIRED DATA ====="
echo

mkdir data data/train data/test data/local data/local/dict
$python3_cmd scripts/prepare_data.py