#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ./main.py \
    --config ./configs/score_flow/face_pytorch_gaussian.py \
    --eval_folder inter \
    --mode inter \
    --workdir '../autodl-tmp/exp/face-condition/'

