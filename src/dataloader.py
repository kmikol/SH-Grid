#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import cv2
import math
import pickle
from tqdm import tqdm
import pandas as pd

try:
    import nvcodec 
    PYNVCODEC_AVAILABLE = True
except ImportError:
    print("PyNvCodec not found")
    PYNVCODEC_AVAILABLE = False


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, path,  device='cpu',gray=True, subsample_factor=None, image_dtype=torch.float32):
        
        self.gray=gray
        self.path = path
        video_file = os.path.join(path,'video.avi')
        self.cache_enabled = False

        self.image_dtype = image_dtype

        # Load poses if possible
        poses_file = os.path.join(path,'poses.pickle')
        if os.path.exists(poses_file):
            with open(poses_file, 'rb') as fp:
                self.poses = pickle.load(fp)
        else:
            self.poses = None

        # Load time if possible
        time_file = os.path.join(path,'time.pickle')
        if os.path.exists(time_file):
            with open(time_file, 'rb') as fp:
                self.time = pickle.load(fp)
        else:
            self.time = None

        # Load transducer object if possible
        transducer_file = os.path.join(path,'transducer_object.pickle')
        if os.path.exists(transducer_file):
            with open(transducer_file, 'rb') as fp:
                self.transducer_object = pickle.load(fp)
        else:
            self.transducer_object = None

        # Load transducer parameters if possible
        transducer_file = os.path.join(path,'transducer_parameters.pickle')
        if os.path.exists(transducer_file):
            with open(transducer_file, 'rb') as fp:
                self.transducer_params = pickle.load(fp)
        else:
            self.transducer_params = None

        # Load the validation indices if available
        validation_indices_file = os.path.join(path,'validation_indices.csv')
        if os.path.exists(validation_indices_file):
            df = pd.read_csv(validation_indices_file)
            self.validation_indices = df['validation_indices'].to_list()

        else:
            self.validation_indices = None


        if device == 'cuda' and PYNVCODEC_AVAILABLE:
            self.decoder = nvcodec.PyNvDecoder(video_file, nvcodec.PixelFormat.RGB_PLANAR, {'gpu_id': 0})
            self.frame_width, self.frame_height = self.decoder.Width(), self.decoder.Height()
            self.num_frames = self.decoder.Frames()
            self.device = 'cuda'
        else:
            if device == 'cuda':
                print("PyNvCodec not available, falling back to CPU decoding followed by transfer to GPU")
                self.device = device
            else:
                self.device = device

            self.decoder = cv2.VideoCapture(video_file)
            self.frame_width, self.frame_height = int(self.decoder.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.decoder.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.num_frames = int(self.decoder.get(cv2.CAP_PROP_FRAME_COUNT))


        if subsample_factor is not None:
            self.subsample_factor = subsample_factor
            self.num_frames = math.floor(self.num_frames/subsample_factor)
            self.poses = self.poses[::subsample_factor]

            if self.time is not None:
                if isinstance(self.time,dict):
                    for k in self.time.keys():
                        if isinstance(self.time[k],int):
                            continue
                        self.time[k] = self.time[k][::subsample_factor]
                else:
                    self.time = self.time[::subsample_factor]
        else:
            self.subsample_factor = None

        print("Video Dataset")
        print(f"Number of frames: {self.num_frames}, height: {self.frame_height}, width: {self.frame_width}\n")



    def decode_gpu(self,idx):
        raw_surface = nvcodec.Surface(self.frame_width, self.frame_height, nvcodec.PixelFormat.RGB_PLANAR, 0)
        success, _ = self.decoder.DecodeSingleFrame(raw_surface, idx)
        if not success:
            raise ValueError(f"Error decoding frame {idx}")

        img = torch.tensor(raw_surface.PlanePtr(), dtype=torch.uint8, device=self.device).view(3, self.frame_height, self.frame_width)
        
        if self.gray:
            img = img.mean(0,keepdim=True)

        if self.image_dtype == torch.float32:
            return img.float() / 255.0
        elif self.image_dtype == torch.uint8:
            return img
        else:
            raise ValueError("Invalid image_dtype")

    def decode_cpu(self, idx):
        self.decoder.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = self.decoder.read()
        if not success:
            raise ValueError(f"Error decoding frame {idx}")

        if self.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[None]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))

        if self.image_dtype == torch.uint8:
            return torch.tensor(frame, dtype=torch.uint8, device=self.device)
        elif self.image_dtype == torch.float32:
            return torch.tensor(frame, dtype=torch.float32, device=self.device) / 255.0
        else:
            raise ValueError("Invalid image_dtype")

    def enable_cache(self):

        channels = 1 if self.gray else 3

        self.image_cache = torch.empty((self.num_frames, channels,
                                        self.frame_height,
                                        self.frame_width),
                                        dtype=self.image_dtype,
                                        device=self.device)

        for i in tqdm(range(self.num_frames), desc="Caching images"):
            data = self.__getitem__(i)
            self.image_cache[i] = data['image']

        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.image_cache = None


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cache_enabled:
            data = {}
            data['image'] = self.image_cache[idx]

            if self.poses is not None:
                data['pose'] = self.poses[idx].to(self.device)

            return data
        else:

            data = {}

            if self.subsample_factor is not None:
                if isinstance(idx,list):
                    idx_img = [int(i*self.subsample_factor) for i in idx]
                elif isinstance(idx,int):
                    idx_img = int(idx*self.subsample_factor)
                else:
                    raise Exception
            else:
                idx_img = idx

            if self.poses is not None:
                data['pose'] = self.poses[idx].to(self.device)

            if self.device == 'cuda' and PYNVCODEC_AVAILABLE:
                img = self.decode_gpu(idx_img)
            else:
                img = self.decode_cpu(idx_img)

            data['image'] = img

        return data
