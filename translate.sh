CUDA_VISIBLE_DEVICES=0 python ./main.py \
     --config ./configs/score_flow/face_pytorch_gaussian.py \
     --eval_folder translation_650k \
     --mode translation \
     --workdir '../autodl-tmp/exp/face-condition/'
