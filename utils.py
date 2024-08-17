from torch.utils.data import Dataset
from pathlib import Path
import PIL
import os
import random 
import numpy as np
import sys
from TeCoA.utils import load_val_datasets, get_text_prompts_val
from torchvision import transforms


class GeneratedDataset(Dataset):
    def __init__(
        self,
        args
    ):  
        #print(args.use_imagenet_origin)
        if args.use_generated_dataset == False:
            print("use imagenet original dataset")
            self.data_root = os.path.join(args.imagenet_root, 'train')
        else:
            print("use generated dataset!")
            self.data_root = Path(args.generated_data_root)
            if not self.data_root.exists():
                raise ValueError("Generated images root doesn't exists.")

        # our
        dataset_name = [args.testdata]
        test_dataset = load_val_datasets(args, dataset_name)[0]
        self.transform = test_dataset.transform

        _ = get_text_prompts_val([test_dataset], dataset_name, template='This is a photo of a {}')[0] # for SUN397
        if hasattr(test_dataset, 'categories'):
            self.class_names = test_dataset.categories # for caltech256
        elif hasattr(test_dataset, '_classes'):
            self.class_names = test_dataset._classes # flowers102
        else: 
            self.class_names = test_dataset.classes
        

        self.files_list = []
        #self.index = []
        self.y = []
        self._length = 0
        for (i, c) in enumerate(self.class_names):
            #n = len(os.listdir(os.path.join(self.generated_data_root, c))) # same value with num shots
            n = args.num_shot
            all_c_files = os.listdir(os.path.join(self.data_root, c))
            selected_c_files = random.sample(all_c_files, n)
            self.files_list.extend(selected_c_files)
            #self.index.extend(range(n)) # to direct instance in same class folder
            self.y.extend(n * [i]) # to represent class label 
            self._length += n

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        image = PIL.Image.open(os.path.join(self.data_root, self.class_names[self.y[index]], self.files_list[index])).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.y[index]
        return image, label
    

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps, min_lr=0.0):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr + min_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def return_prompt(dataset_name, personalized=False):
    # we refer to the prompt template in https://github.com/cvlab-columbia/ZSRobust4FoundationModel
    if dataset_name in ['StanfordCars', 'Caltech256']:
        template =  'A photo of a {}.'
    elif dataset_name == 'flowers102':
        template = 'A photo of a {}, a type of flower.'
    elif dataset_name == 'Food101':
        template = 'A photo of a {}, a type of food.'
    elif dataset_name == 'oxfordpet':
        template = 'A photo of a {}, a type of pet.'
    elif dataset_name == 'dtd':
        template = 'A surface with a {} texture.'
    elif dataset_name == 'EuroSAT':
        template = "A centered satellite photo of {}."
    elif dataset_name == 'cropdisease':
        template = 'A leaf photo of {}.'
    elif dataset_name == 'isic2018':
        template = 'A skin photo of {}.'
    else: # imagenet, sun, stl-10
        template = 'This is a photo of a {}.'
    
    if personalized:
        return template.format('sks')
    else:
        return template