#!/bin/bash
set -x
set -e

NET=$1
DATA=$2
CKPT=$3
MODEL=$3
MODEL=`echo ${MODEL##*\/} | cut -f 1 -d '.'`
EXP_DIR=$3
EXP_DIR=${CKPT%model_*.pth}

if [ $DATA = "CUB" ]
then
  DATA_P="datasets/CUB"
else
  DATA_P="datasets/IMNET"
fi

LOG="${EXP_DIR}test_${MODEL}_`date +'%Y-%m-%d_%H-%M-%S'`.log"

echo Logging output to "$LOG"

/path/to/python3 main.py --eval --model ${NET} --data-path ${DATA_P} --data-set ${DATA} --resume ${CKPT} | tee "${LOG}"

