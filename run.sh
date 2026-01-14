#!/bin/bash
DATASET_NAME="CUHK-PEDES"
#DATASET_NAME="ICFG-PEDES"
#DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name iira \
--img_aug \
--batch_size 32 \
--m_ratio 0.15 \
--confidence 0.15 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'ccm+cdl' \
--num_epoch 60
#echo -e '\n\n\n'

