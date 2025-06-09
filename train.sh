
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./main.py \
     --config ./configs/score_flow/face_pytorch_gaussian.py \
     --eval_folder eval \
     --mode train \
     --workdir '../autodl-tmp/exp/face-condition'