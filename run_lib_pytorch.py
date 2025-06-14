# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
#import ReFlow.losses as losses
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import pdb

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds = datasets.get_pytorch_dataset(config)

  train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4)
  eval_ds_loader = torch.utils.data.DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  step = initial_step - 1
  while step <= (num_train_steps + 1):
    for _, (image, label) in enumerate(train_ds_loader):
        step += 1
        batch = image.to(config.device).float()
        batch = scaler(batch)
        label = label.to(config.device).float()
    
        # Execute one training step
        loss = train_step_fn(state, batch, label)

        if step % config.training.log_freq == 0:
          logging.info("step: %d, training_loss: %.5e, reflow_loss: %.5e" % (step, loss[0].item(), \
                                                                                         loss[1].item()))
          writer.add_scalar("training/loss", loss[0].item(), step)
          writer.add_scalar("training/reflow_loss", loss[1].item(), step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
          save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
          for eval_image, eval_label in eval_ds_loader:  
            break  
          eval_batch = eval_image.to(config.device).float()
          eval_batch = scaler(eval_batch)
          eval_label = eval_label.to(config.device).float()
          eval_loss = eval_step_fn(state, eval_batch, eval_label)
          logging.info("step: %d, eval_loss: %.5e, reflow_loss: %.5e" % (step, eval_loss[0].item(),\
                                                                                      eval_loss[1].item()))
          writer.add_scalar("eval/loss", eval_loss[0].item(), step)
          writer.add_scalar("eval/reflow_loss", eval_loss[1].item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
          # Save the checkpoint.
          save_step = step // config.training.snapshot_freq
          save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

          # Generate and save samples
          if config.training.snapshot_sampling:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            #nrow = int(np.sqrt(sampling_shape[0]))
            nrow = 10 # modified
            # face datasets includes 2 classes
            context = [i % 2 for i in range(6)] * nrow
            context = torch.tensor(context).to(config.device).float()
            sample, n = sampling_fn(score_model, context_condition=context)
            ema.restore(score_model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            tf.io.gfile.makedirs(this_sample_dir)
            #nrow = int(np.sqrt(sample.shape[0]))
            # nrow: number of images displayed in each row.
            image_grid = make_grid(sample, nrow, padding=2)
            sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
              np.save(fout, sample)

            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
              save_image(image_grid, fout)



def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            uniform_dequantization=config.data.uniform_dequantization,
  #                                            evaluation=True)
  train_ds, eval_ds = datasets.get_pytorch_dataset(config)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)


  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  #train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
  #                                                    uniform_dequantization=True, evaluation=True)
  train_ds_bpd, eval_ds_bpd = datasets.get_pytorch_dataset(config)   ###NOTE: XC: fix later

  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if (config.eval.enable_sampling) or (config.eval.enable_figures_only):
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())
    
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}")
        tf.io.gfile.makedirs(this_sample_dir)
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model,
                                                       inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        gc.collect()
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(
            io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
          fout.write(io_buffer.getvalue())

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
      for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
            all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      data_pools = data_stats["pool_3"]

      # Compute FID/KID/IS on all samples together.
      if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
      else:
        inception_score = -1

      fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
      # Hack to get tfgan KID work for eager execution.
      tf_data_pools = tf.convert_to_tensor(data_pools)
      tf_all_pools = tf.convert_to_tensor(all_pools)
      kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
      del tf_data_pools, tf_all_pools

      logging.info(
        "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
          ckpt, inception_score, fid, kid))

      with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                             "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
        f.write(io_buffer.getvalue())
   

    if config.eval.enable_figures_only:
      import torchvision
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        logging.info("sampling only figures -- ckpt: %d, round: %d" % (ckpt, r))
        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
        
        # Directory to save samples. Different for each host to avoid writing conflicts
        samples, n = sampling_fn(score_model)
        torchvision.utils.save_image(samples.clamp_(0.0, 1.0), os.path.join(this_sample_dir, '%d.png'%r), nrow=10, normalize=False)

def img_translation(config,
                    workdir,
                    cond_lst=[0, 1],
                    eval_folder="translation"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    cond_lst: Classes for translation, cond_lst[0] to cond_lst[1]
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            uniform_dequantization=config.data.uniform_dequantization,
  #                                            evaluation=True)
  train_ds, eval_ds = datasets.get_pytorch_dataset(config, cond_lst[0])
  train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
  eval_ds_loader = torch.utils.data.DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model_1 = mutils.create_model(config)
  optimizer_1 = losses.get_optimizer(config, score_model_1.parameters())
  ema_1 = ExponentialMovingAverage(score_model_1.parameters(), decay=config.model.ema_rate)
  state_1 = dict(optimizer=optimizer_1, model=score_model_1, ema=ema_1, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, translation=True)

  
  #ckpt_filename_cat = os.path.join(checkpoint_dir, "cat_checkpoint_10.pth")
  #ckpt_filename_face = os.path.join(checkpoint_dir, "face_checkpoint_10.pth")
  #ckpt_filename_cat = os.path.join('./exp_cat/translate/checkpoints', "cat_checkpoint_10.pth")
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_13.pth")

  state_1 = restore_checkpoint(ckpt_filename, state_1, device=config.device)
  ema_1.copy_to(score_model_1.parameters()) 

  import torchvision
  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
  for r in range(num_sampling_rounds):
    z0 = next(iter(train_ds_loader))[:config.eval.batch_size, ...]
    z0 = scaler(z0)
    #z0 = None
    logging.info("sampling translation figures")
    this_sample_dir = os.path.join(eval_dir)
    
    # Directory to save samples. Different for each host to avoid writing conflicts
    samples, n = sampling_fn(score_model_1, cond_lst, z0)
    print(f'saving imgs round {r}')
    for i, sample in enumerate(samples):
      #torchvision.utils.save_image(sample.clamp_(0.0, 1.0), os.path.join(this_sample_dir, f'translate_{r}_{0.1 + i * 0.2}.png'), nrow=1, normalize=False)
      torchvision.utils.save_image(sample.clamp_(0.0, 1.0), os.path.join(this_sample_dir, f'translate_{r}.png'), nrow=4, normalize=False)    
    torchvision.utils.save_image(inverse_scaler(z0).clamp_(0.0, 1.0), os.path.join(this_sample_dir, f'source_{r}.png'), nrow=4, normalize=False)


def gen_translate(config,
                    workdir,
                    cond_lst=[0, 1],
                    eval_folder="translation"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    cond_lst: Classes for translation, cond_lst[0] to cond_lst[1]
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model_1 = mutils.create_model(config)
  optimizer_1 = losses.get_optimizer(config, score_model_1.parameters())
  ema_1 = ExponentialMovingAverage(score_model_1.parameters(), decay=config.model.ema_rate)
  state_1 = dict(optimizer=optimizer_1, model=score_model_1, ema=ema_1, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn_gen = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  sampling_fn_trans = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, translation=True)

  
  #ckpt_filename_cat = os.path.join(checkpoint_dir, "cat_checkpoint_10.pth")
  #ckpt_filename_face = os.path.join(checkpoint_dir, "face_checkpoint_10.pth")
  #ckpt_filename_cat = os.path.join('./exp_cat/translate/checkpoints', "cat_checkpoint_10.pth")
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_13.pth")

  state_1 = restore_checkpoint(ckpt_filename, state_1, device=config.device)
  ema_1.copy_to(score_model_1.parameters()) 

  import torchvision
  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
  context_s = torch.ones(sampling_shape[0]) * cond_lst[0]
  context_s = context_s.to(config.device)
  context_t = torch.ones(sampling_shape[0]) * cond_lst[1]
  context_t = context_t.to(config.device)
  for r in range(num_sampling_rounds):
    # get the Gaussian nosie
    z0 = sde.get_z0(torch.zeros(sampling_shape, device=config.device), train=False).to(config.device)
    z0 = z0.detach().clone()
    sample_s, n = sampling_fn_gen(score_model_1, context_condition=context_s, z=z0)
    sample_t, n = sampling_fn_gen(score_model_1, context_condition=context_t, z=z0)
    sample_s = scaler(sample_s)
    #z0 = None
    logging.info("sampling translation figures")
    
    samples, n = sampling_fn_trans(score_model_1, cond_lst, sample_s)
    print(f'saving imgs round {r}')
    for i, sample in enumerate(samples):
      #torchvision.utils.save_image(sample.clamp_(0.0, 1.0), os.path.join(this_sample_dir, f'translate_{r}_{0.1 + i * 0.2}.png'), nrow=1, normalize=False)
      torchvision.utils.save_image(sample.clamp_(0.0, 1.0), os.path.join(eval_dir, f'translate_{r}.png'), nrow=4, normalize=False)    
    torchvision.utils.save_image(inverse_scaler(sample_s).clamp_(0.0, 1.0), os.path.join(eval_dir, f'source_{r}.png'), nrow=4, normalize=False)
    torchvision.utils.save_image(sample_t.clamp_(0.0, 1.0), os.path.join(eval_dir, f'target_{r}.png'), nrow=4, normalize=False)
    for i in range(sampling_shape[0]):
      trans_path = os.path.join(eval_dir, f'trans_{r}')
      source_path = os.path.join(eval_dir, f'source_{r}')
      target_path = os.path.join(eval_dir, f'target_{r}')
      if not os.path.exists(trans_path):
        tf.io.gfile.makedirs(trans_path)
      if not os.path.exists(source_path):
        tf.io.gfile.makedirs(source_path)
      if not os.path.exists(target_path):
        tf.io.gfile.makedirs(target_path)
      torchvision.utils.save_image(samples[-1][i].clamp_(0.0, 1.0), os.path.join(trans_path, f'{i}.png'), nrow=4, normalize=False)    
      torchvision.utils.save_image(inverse_scaler(sample_s[i]).clamp_(0.0, 1.0), os.path.join(source_path, f'{i}.png'), nrow=4, normalize=False)
      torchvision.utils.save_image(sample_t[i].clamp_(0.0, 1.0), os.path.join(target_path, f'{i}.png'), nrow=4, normalize=False)

def interpolation(config,
                    workdir,
                    cond_lst = [0, 1],
                    eval_folder="interpolation"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    cond_lst: Classes for translation, cond_lst[0] to cond_lst[1]
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model_1 = mutils.create_model(config)
  optimizer_1 = losses.get_optimizer(config, score_model_1.parameters())
  ema_1 = ExponentialMovingAverage(score_model_1.parameters(), decay=config.model.ema_rate)
  state_1 = dict(optimizer=optimizer_1, model=score_model_1, ema=ema_1, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  sampling_shape = (1,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  
  #ckpt_filename_cat = os.path.join(checkpoint_dir, "cat_checkpoint_10.pth")
  #ckpt_filename_face = os.path.join(checkpoint_dir, "face_checkpoint_10.pth")
  #ckpt_filename_cat = os.path.join('./exp_cat/translate/checkpoints', "cat_checkpoint_10.pth")
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_13.pth")

  state_1 = restore_checkpoint(ckpt_filename, state_1, device=config.device)
  ema_1.copy_to(score_model_1.parameters()) 

  import torchvision
  num_sampling_rounds = config.eval.num_samples + 1
  context_1 = torch.ones(sampling_shape[0]) * cond_lst[0]
  context_1 = context_1.to(config.device)
  context_2 = torch.ones(sampling_shape[0]) * cond_lst[1]
  context_2 = context_2.to(config.device)
  for r in range(num_sampling_rounds):
    z0 = sde.get_z0(torch.zeros(sampling_shape, device=config.device), train=False).to(config.device)
    z0 = z0.detach().clone()
    z1 = sde.get_z0(torch.zeros(sampling_shape, device=config.device), train=False).to(config.device)
    z1 = z1.detach().clone()
    #z0 = None
    logging.info("sampling interpolation figures")
    res_1 = []
    res_2 = []
    for alpha in np.arange(0.1, 1.0, 0.1):
      z = np.sqrt(alpha) * z0 + np.sqrt(1 - alpha) * z1
      #z = alpha * z0 + (1 - alpha) * z1
      sample_1, n = sampling_fn(score_model_1, context_condition=context_1, z=z)
      sample_2, n = sampling_fn(score_model_1, context_condition=context_2, z=z)
      res_1.append(sample_1)
      res_2.append(sample_2)
    res_1 = torch.cat(res_1, dim=0).to(config.device)
    res_2 = torch.cat(res_2, dim=0).to(config.device)
    torchvision.utils.save_image(res_1.clamp_(0.0, 1.0), os.path.join(eval_dir, f'res1_{r}.png'), nrow=9, normalize=False)
    torchvision.utils.save_image(res_2.clamp_(0.0, 1.0), os.path.join(eval_dir, f'res2_{r}.png'), nrow=9, normalize=False)



'''
def img_translation(config,
                    workdir,
                    eval_folder="translation"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  #train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                            uniform_dequantization=config.data.uniform_dequantization,
  #                                            evaluation=True)
  train_ds, eval_ds = datasets.get_pytorch_dataset(config)
  train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
  eval_ds_loader = torch.utils.data.DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model_1 = mutils.create_model(config)
  optimizer_1 = losses.get_optimizer(config, score_model_1.parameters())
  ema_1 = ExponentialMovingAverage(score_model_1.parameters(), decay=config.model.ema_rate)
  state_1 = dict(optimizer=optimizer_1, model=score_model_1, ema=ema_1, step=0)

  score_model_2 = mutils.create_model(config)
  optimizer_2 = losses.get_optimizer(config, score_model_2.parameters())
  ema_2 = ExponentialMovingAverage(score_model_2.parameters(), decay=config.model.ema_rate)
  state_2 = dict(optimizer=optimizer_2, model=score_model_2, ema=ema_2, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'rectified_flow':
    sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type, noise_scale=config.sampling.init_noise_scale, use_ode_sampler=config.sampling.use_ode_sampler, sigma_var=config.sampling.sigma_variance, ode_tol=config.sampling.ode_tol, sample_N=config.sampling.sample_N)
    sampling_eps = 1e-3
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, translation=True)

  
  #ckpt_filename_cat = os.path.join(checkpoint_dir, "cat_checkpoint_10.pth")
  #ckpt_filename_face = os.path.join(checkpoint_dir, "face_checkpoint_10.pth")
  #ckpt_filename_cat = os.path.join('./exp_cat/translate/checkpoints', "cat_checkpoint_10.pth")
  ckpt_filename_cat = os.path.join('./exp_cat/exp1/checkpoints', "checkpoint_8.pth")
  ckpt_filename_face = os.path.join('./exp_wild/exp1/checkpoints', "checkpoint_15.pth")

  state_1 = restore_checkpoint(ckpt_filename_cat, state_1, device=config.device)
  state_2 = restore_checkpoint(ckpt_filename_face, state_2, device=config.device)
  ema_1.copy_to(score_model_1.parameters()) 
  ema_2.copy_to(score_model_2.parameters()) 

  import torchvision
  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
  for r in range(num_sampling_rounds):
    z0 = next(iter(train_ds_loader))[:config.eval.batch_size, ...]
    z0 = scaler(z0)
    #z0 = None
    logging.info("sampling translation figures")
    this_sample_dir = os.path.join(eval_dir)
    
    # Directory to save samples. Different for each host to avoid writing conflicts
    samples, n = sampling_fn(score_model_1, score_model_2, z0)
    print(f'saving imgs round {r}')
    for i, sample in enumerate(samples):
      #torchvision.utils.save_image(sample.clamp_(0.0, 1.0), os.path.join(this_sample_dir, f'translate_{r}_{0.1 + i * 0.2}.png'), nrow=1, normalize=False)
      torchvision.utils.save_image(sample.clamp_(0.0, 1.0), os.path.join(this_sample_dir, f'translate_{r}.png'), nrow=1, normalize=False)    
    torchvision.utils.save_image(inverse_scaler(z0).clamp_(0.0, 1.0), os.path.join(this_sample_dir, f'source_{r}.png'), nrow=1, normalize=False)
'''
