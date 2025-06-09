#!/bin/bash
num_runs=5

#for ((i=1; i<=num_runs; i++))
#do
#    CUDA_VISIBLE_DEVICES=0,1 python ./main.py \
#     --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py \
#     --eval_folder eval_$i \
#     --mode eval \
#     --workdir '/root/autodl-tmp/linear_wieght_t-b128'
#done

for ((i=1; i<=num_runs; i++))
do
    CUDA_VISIBLE_DEVICES=0 python ./main.py \
     --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py \
     --eval_folder eval_new_$i \
     --mode eval \
     --workdir './exp/conditional_reflow'
done
