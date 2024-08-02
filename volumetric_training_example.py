# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: © 2024 Tyler Dodds

# Example training denoising diffusion using volumetric .nii dataset.

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_3d import Unet3D, GaussianDiffusion3D, Trainer3D
from denoising_diffusion_pytorch.volumetric_data import NiiDataset, NiiSaver

import torch

dataset_folder = None #Set path to data folder
load_milestone = None

normalize_from = (0, 480)
#NB We must rescale to a relatively small volume size when GPU memory is scarce.
crop_to = (64, 64, 64)
volume_size = (32, 32, 32)
channels = 1

batch_size = 1
gradient_accumulate_every = 16      # NB Need to get an 'effective' batch size (gradient_accumulate_every * train_batch_size) of 16, and we only have a batch size of 1
timesteps = 100
train_num_steps = 2100
save_and_sample_every = 210

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = NiiDataset(dataset_folder, crop_to_min_shape = False, crop_to = crop_to, normalize_from = normalize_from, resize_to = volume_size)
    saver = NiiSaver()

    model = Unet3D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = channels,
        flash_attn = True
    ).to(device)

    diffusion = GaussianDiffusion3D(
        model,
        image_size = volume_size,
        timesteps = timesteps,
        sampling_timesteps = 25,
    ).to(device)

    trainer = Trainer3D(
        diffusion,
        dataset,
        saver,
        train_batch_size = batch_size,
        train_lr = 8e-5,
        train_num_steps = train_num_steps,
        gradient_accumulate_every = gradient_accumulate_every,  
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        num_samples = 5,
        save_and_sample_every = save_and_sample_every,
        save_best_and_latest_only = True,
    )

    if load_milestone is not None:
        trainer.load(load_milestone)
    
    trainer.train()
