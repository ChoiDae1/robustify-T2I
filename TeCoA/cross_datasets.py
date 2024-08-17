import os
import json
import random
from collections import defaultdict

import pandas as pd
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torch
import torch.nn as nn
import torchvision.datasets as D


class ISIC2018(torch.utils.data.Dataset):
    dataset_dir = 'ISIC2018'
    def __init__(self, root, transform, prompt_template= 'A photo of {}.'):
        super().__init__()
        root = os.path.join(root, self.dataset_dir)
        self.img_path = os.path.join(root, 'ISIC2018_Task3_Training_Input')
        target_file = os.path.join(root, 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
        self.classes = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion']
        self.data_info = pd.read_csv(target_file, skiprows=[0], header=None)
        self.image_names = np.asarray(self.data_info.iloc[:, 0])
        self.targets = np.asarray(self.data_info.iloc[:, 1:])
        self.targets = (self.targets != 0).argmax(axis=1)
        #print(f'minimum {max(self.targets)}')
        self.transform = transform
        self.prompt_template = prompt_template
        self.clip_prompts = [ 
            prompt_template.format(label.lower().replace('_', ' ').replace('-', ' ')) \
            for label in self.classes
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        filename = os.path.join(self.img_path, self.image_names[index] + ".jpg")
        img = Image.open(filename, mode='r').convert("RGB")
        target = self.targets[index]
        return self.transform(img), target


class ChestX(torch.utils.data.Dataset):
    dataset_dir = 'chest-x'
    def __init__(self, root, transform, prompt_template= 'A chest x-ray of {}.'): #chest x-ray 
        super().__init__()
        root = os.path.join(root, self.dataset_dir)
        self.img_path = os.path.join(root, 'images')
        target_file = os.path.join(root, 'Data_Entry_2017.csv')

        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        labels_set = []

        self.data_info = pd.read_csv(target_file, skiprows=[0], header=None)
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.targets_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_names  = []
        self.targets = []
        for name, label in zip(self.image_name_all, self.targets_all):
            label = label.split("|")
            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.targets.append(self.labels_maps[label[0]])
                self.image_names.append(name)
        self.image_names = np.asarray(self.image_names)
        self.targets = np.asarray(self.targets)
        self.transform = transform
        
        self.classes = self.used_labels
        self.prompt_template = prompt_template
        self.clip_prompts = [ 
            prompt_template.format(label.lower().replace('_', ' ').replace('-', ' ')) \
            for label in self.classes
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        filename = os.path.join(self.img_path, self.image_names[index])
        img = Image.open(filename, mode='r').convert("RGB")
        target = self.targets[index]
        return self.transform(img), target


class CropDisease(D.ImageFolder):
    dataset_dir = 'crop-disease'
    def __init__(self, root, transform, prompt_template= 'A leaf photo of {}.'):
        root = os.path.join(root, self.dataset_dir, 'train')
        super().__init__(root, transform)
        self.classes = ['Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Apple healthy', 'Blueberry healthy', 'Cherry Powdery Mildew', 'Cherry healthy', 'Corn Gray Leaf Spot',
                        'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black Rot', 'Grape Black Measles (Esca)','Grape Leaf Blight','Grape Healthy',
                        'Orange Huanglongbing (Citrus Greening)', 'Peach Bacterial Spot', 'Peach healthy', 'Bell Pepper Bacterial Spot',
                        'Bell Pepper healthy', 'Potato Early Blight', 'Potato Late Blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery Mildew',
                        'Strawberry Leaf Scorch', 'Strawberry Healthy', 'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
                        'Tomato Two Spotted Spider Mite', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus','Tomato healthy'] # 38
        assert len(self.classes) == 38
        self.prompt_template = prompt_template
        self.clip_prompts = [ 
            prompt_template.format(label.lower().replace('_', ' ').replace('-', ' ')) \
            for label in self.classes
        ]
