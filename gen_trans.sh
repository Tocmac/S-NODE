CUDA_VISIBLE_DEVICES=0 python ./main.py \
     --config ./configs/score_flow/face_pytorch_gaussian.py \
     --eval_folder s_t_trans_650k \
     --mode gen_trans \
     --workdir '../autodl-tmp/exp/face-condition/'
