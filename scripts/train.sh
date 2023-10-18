#!/bin/bash
set -x
set -e

NET=$1
DATA=$2
EPOCH=$3
EXP_DIR=$4
POINT="/path/to/weights/deit_small_patch16_224-cd65a155.pth"

if [ ${DATA} = "CUB" ]
then
  BATCH=128
  DATA_P="datasets/CUB"
  LR=5e-5
else
  BATCH=128
  DATA_P="datasets/IMNET"
  LR=5e-4
fi


LOG="${EXP_DIR}/train_`date +'%Y-%m-%d_%H-%M-%S'`.log"

echo Logging output to "$LOG"

if [ ! -d "${EXP_DIR}" ]
then
  mkdir -p "${EXP_DIR}"
else
  read -p "The path exists, DELETE? (y/n)" DE
  case ${DE} in
    Y | y)
      rm -rf "${EXP_DIR}"
      mkdir -p "${EXP_DIR}";;
    *)
      exit
  esac
fi

/path/to/torchrun --nproc_per_node=4 --master_port=11240 main.py --model ${NET} --batch-size ${BATCH} --data-path ${DATA_P} --epochs ${EPOCH} --data-set ${DATA} --lr ${LR} --output_dir ${EXP_DIR} --resume ${POINT} | tee ${LOG}
