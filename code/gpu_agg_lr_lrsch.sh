#!/user/bin/env bash
gpu=$1
agg=$2
lr=$3
lrsch=$4

CUDA_VISIBLE_DEVICES=$gpu nohup python cli.py train --aggregation $agg --learning_rate $lr --lrschedule $lrsch > data/log/twopunch_${agg}_${lr}_${lrsch}.out &
