# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: © 2024 Tyler Dodds

from abc import ABC, abstractmethod
from glob import glob
import nibabel
import numpy as np
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class VolumetricSaver(ABC):
    @abstractmethod
    def save(self, folder : str, filename : str, data : np.ndarray):
        """
        Saves three-dimensional volumetric data in the given folder with the given filename.
        """

class NiiSaver(VolumetricSaver):
    def __init__(self, remap_to = None):
        self.remap_to = remap_to

    def save(self, folder : str, filename : str, data : np.ndarray):
        if self.remap_to is not None:
            data = np.interp(data, (0, 1), self.remap_to)
        path = os.path.join(folder, f"{filename}.nii.gz")
        nibabel_image = nibabel.Nifti1Image(data, np.eye(4), nibabel.Nifti1Header())
        nibabel.save(nibabel_image, path)

class NiiDataset(Dataset):
    def get_nii_image_paths(folder : str, extensions : List[str]):
        images = []
        for extension in extensions:
            images = images + glob(f"{folder}/*.{extension}")
        return images
    
    def __init__(self, nii_dir, crop_to_min_shape, crop_to = None, normalize_from = None, resize_to = None, transform = None):
        self.nii_dir = nii_dir
        self.crop_to_min_shape = crop_to_min_shape
        self.crop_to = crop_to
        self.normalize_from = normalize_from
        if self.normalize_from is None:
            self.normalize_type = 0
        elif self.normalize_from == 'each':
            self.normalize_type = 1
        elif type(self.normalize_from) is tuple or type(self.normalize_from) is list:
            self.normalize_type = 2
        else:
            print("Expect normalize_from to be a list, tuple, or 'each'")
            self.normalize_type = 0
        self.resize_to = resize_to
        self.transform = transform
        self.nii_paths = NiiDataset.get_nii_image_paths(self.nii_dir, ["nii", "nii.gz"])
        self.min_shape = None
        self.nii_shapes = [nibabel.load(path).shape for path in self.nii_paths]
        self.min_shape = np.array(self.nii_shapes).min(axis = 0)
    
    def _get_min_shape(self) -> np.ndarray:
        if self.min_shape is None:
            self.nii_shapes = [nibabel.load(path).shape for path in self.nii_paths]
            self.min_shape = np.array(self.nii_shapes).min(axis = 0)
        return self.min_shape

    def __len__(self):
        return len(self.nii_paths)

    def __getitem__(self, idx):
        img_path = self.nii_paths[idx]
        image = nibabel.load(img_path)
        data = image.get_fdata()
        if self.crop_to_min_shape or self.crop_to is not None:
            crop_shape = None
            if self.crop_to_min_shape:
                crop_shape = self._get_min_shape()
            if self.crop_to is not None:
                crop_array = np.array(self.crop_to)
                if crop_shape is None:
                    crop_shape = crop_array
                else:
                    crop_shape = np.minimum(crop_shape, crop_array)
            shape = data.shape
            shape_excess = shape - crop_shape
            shape_offset = shape_excess // 2
            shape_end = shape_offset + crop_shape
            slices = tuple(map(slice, shape_offset, shape_end))
            data = data[slices]
        if self.normalize_type != 0:
            if self.normalize_type == 1:
                d_min = np.min(data)
                d_max = np.max(data)
                data = np.interp(data, (d_min, d_max), (0, 1))
            elif self.normalize_type == 2:
                data = np.interp(data, self.normalize_from, (0, 1))
            data = np.clip(data, 0, 1)
        if self.transform:
            data = self.transform(data)
        if self.resize_to:
            data = torch.tensor(data[np.newaxis, np.newaxis, ...], dtype = torch.float32)#Dimensions are batch_size, channels, d1 through d3; needed for F.interpolate
            data = F.interpolate(data, self.resize_to).squeeze()
        else:
            data = torch.tensor(data)
        data.unsqueeze_(0)#Unsqueeze a length-1 'channel' dimension at the start
        return data
