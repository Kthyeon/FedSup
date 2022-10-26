#!/bin/bash
betalist="0.01 1.0"
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
                                    for beta in $betalist
                                    do
                                        python main.py  --beta $beta \
                                                        --dataset cifar10 \
                                                        --gpu 0 \
                                                        --frac $frac \
                                                        --label_smoothing $lsm \
                                                        --local_ep $le \
                                                        --momentum $mt \
                                                        --inplace_distill $indist \
                                                        --num_arch_training $numarchtraining \
                                                        --drop_connect $dropconnect \
                                                        --dropout $dropout \
                                                        --results_save $seed \
                                                        --diri "True" \
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