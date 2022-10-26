#!/bin/bash
shardlist="2 5 10"
lelist="5"
lslist="0.0"
mtlist="0.5"
indistill="False"
numarchtraininglist="1"
dropconnectlist="0.0"
dropoutlist="0.0"
seedlist="0"
fraclist="0.1 1.0"

for frac in $fraclist
do
    for seed in $seedlist
    do
        for dropout in $dropoutlist
        do
            for dropconnect in $dropconnectlist
            do
                for numarchtraining in $numarchtraininglist
                do
                    for indist in $indistill
                    do
                        for mt in $mtlist
                        do
                            for lsm in $lslist
                            do
                                for le in $lelist
                                do
                                    for shard in $shardlist
                                    do
                                        python main.py  --shard_per_user $shard \
                                                        --dataset cifar10 \
                                                        --label_smoothing $lsm \
                                                        --gpu 0 \
                                                        --frac $frac \
                                                        --local_ep $le \
                                                        --momentum $mt \
                                                        --inplace_distill $indist \
                                                        --num_arch_training $numarchtraining \
                                                        --drop_connect $dropconnect \
                                                        --dropout $dropout \
                                                        --results_save $seed \
                                                        --diri "False" \
                                                        --config-file config/efficient_fedsup.yml
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done