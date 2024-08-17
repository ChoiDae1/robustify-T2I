import torch.nn as nn
import torch
from typing import *
import clip
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm 
from torch.utils.data import DataLoader
import os
from math import ceil
'''
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NormalizeLayer(torch.nn.Module):
    def __init__(self, means: List[float], sds: List[float]):
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means)
        self.sds = torch.tensor(sds)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        device_ordinal = input.get_device()
        #print(device_ordinal)
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(f'cuda:{device_ordinal}')
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).to(f'cuda:{device_ordinal}')
        return (input - means) / sds

class CLIP_ZeroShot(nn.Module):
    def __init__(self, clip_model, text_list, n_px=(224, 224)):
        super().__init__() 
        self.clip_model = clip_model
        self.text_list = text_list # 하나의 dataset에 대한 text_list 받음
        self.n_px = n_px
        self.normlayer = NormalizeLayer(means=[0.48145466, 0.4578275, 0.40821073], sds=[0.26862954, 0.26130258, 0.27577711])
        # calculate features
        self.set_textfeatures(self.text_list)

    def set_textfeatures(self, text_list):
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(text) for text in text_list]).cuda()
            self.text_features = self.clip_model.encode_text(text_inputs)
            self.text_features = self.text_features/self.text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp().cuda() # exp(log(100)) = 100
            self.text_features = logit_scale * self.text_features

    def forward(self, images):
        images = torch.nn.functional.interpolate(images, self.n_px, mode='bicubic')
        self.text_features = self.text_features.to(images.device)
        images = self.normlayer(images)
        images_features = self.clip_model.encode_image(images)
        images_features = images_features/images_features.norm(dim=-1, keepdim=True)
        logits_per_image = images_features @ self.text_features.T

        return logits_per_image


class CLIPLoss_CE(nn.Module):
    def __init__(self, clip_model, text_list, accelerator, n_px=(224, 224), use_norm=False):
        super().__init__() 
        self.clip_model = clip_model
        self.clip_model.eval().to(accelerator.device)
        self.text_list = text_list
        self.accelerator = accelerator
        self.device = accelerator.device
        self.n_px = n_px
        self.use_norm = use_norm
        if self.use_norm:
           self.normlayer = NormalizeLayer(means=[0.48145466, 0.4578275, 0.40821073], sds=[0.26862954, 0.26130258, 0.27577711])
        self.set_textfeatures(self.text_list)

    def set_textfeatures(self, text_list):
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(text) for text in text_list]).cuda()
            self.text_features = self.clip_model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp().cuda() # exp(log(100)) = 100
            self.text_features = logit_scale * self.text_features

    def forward(self, images, target): # images: (B, C, H, W) texts: list 
        images = torch.nn.functional.interpolate(images, self.n_px, mode='bicubic')
        if self.use_norm:
            images = self.normlayer(images)
        image_features = self.clip_model.encode_image(images)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = image_features @ self.text_features.T
        target = target.to(self.accelerator.device)
        clip_loss = F.cross_entropy(logits_per_image, target)
        
        return clip_loss