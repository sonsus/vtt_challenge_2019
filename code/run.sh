#!/user/bin/env bash
gpu=0

for lr in 3e-3 3e-4;do
    for agg in cat sum elemwise;do
        bash gpu_agg_lr_lrsch.sh $gpu $agg $lr rop &
        sleep 1m 

        gpu=$(($gpu + 1))
    done
done
