# S-NODE

### Dependencies
Run the following commands to install the dependencies:
```
conda create -n rectflow python=3.8
conda activate rectflow
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorflow==2.9.0 tensorflow-probability==0.12.2 tensorflow-gan==2.0.0 tensorflow-datasets==4.6.0
pip install jax==0.4.6 jaxlib==0.4.6 scipy==1.10
pip install numpy==1.21.6 ninja==1.11.1 matplotlib==3.7.0 ml_collections==0.1.1
pip install scikit-image=0.20 lmdb opencv-python
```

### Train
Run the following command to train S-NODE from scratch:
```
python ./main.py --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py --eval_folder eval --mode train --workdir ./logs/S-NODE
```

* ```--config``` The configuration file for this run.

* ```--eval_folder``` The generated images and other files for each evaluation during training will be stroed in ```./workdir/eval_folder```. In this command, it is ```./logs/S-NODE```

* ```---mode``` Mode selection for ```main.py```. Select from ```train```, ```eval```, ```translation```, ```gen_trans``` and ```inter```.


### Sampling and Evaluation

We follow the evaluation pipeline as in [Score SDE](https://github.com/yang-song/score_sde_pytorch). You can download [`cifar10_stats.npz`](https://drive.google.com/file/d/14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI/view?usp=sharing) and save it to `assets/stats/`. 
Then run

```
python ./main.py --config ./configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py --eval_folder eval --mode eval --workdir ./logs/1_rectified_flow --config.eval.enable_sampling  --config.eval.batch_size 1024 --config.eval.num_samples 50000 --config.eval.begin_ckpt 2
```

which uses a batch size of 1024 to sample 50000 images, starting from checkpoint-2.pth, and computes the FID and IS.

### Translation
```
python ./main.py --config ./configs/score_flow/face_pytorch_gaussian.py --eval_folder translation --mode translation --workdir './logs/S-NODE'
```
```
python ./main.py --config ./configs/score_flow/face_pytorch_gaussian.py --eval_folder translation --mode gen_trans --workdir './logs/S-NODE'
```


### Interpolation

```
python ./main.py --config ./configs/score_flow/face_pytorch_gaussian.py --eval_folder inter --mode inter --workdir './logs/S-NODE'
```

## Thanks
A Large portion of this codebase is built upon [RectFlow](https://github.com/gnobitab/RectifiedFlow).