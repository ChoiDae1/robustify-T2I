import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import clip
from utils import *

import ignite.distributed as idist
from TeCoA.utils import load_val_datasets, get_text_prompts_val
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil



def get_arguments():
    parser = argparse.ArgumentParser('Generate CLIP few-training data')
    # dataset 
    parser.add_argument('--root', type=str, default='./datasets/DATA',
                        help='dataset root')
    parser.add_argument('--testdata', type=str, choices=['STL10', 'SUN397','StanfordCars', 'Food101',
                                                         'oxfordpet', 'Caltech256', 'flowers102',
                                                         'dtd','ImageNet','isic', 'EuroSAT', 'cropdisease'], help='test datasetname for generating data')
    parser.add_argument('--use_clip_official', type=bool, default=False)
    # save for generated dataset
    parser.add_argument('--savedir', type=str, default='./datasets/generated_DATA', help='path for saving generated_data')
    parser.add_argument('--num_shot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)

    args = parser.parse_args()

    return args

def main():
    args = get_arguments()
    args.num_workers = 0 
    if args.num_shot > 1:
        args.guidance_scale = 4.0
    else:
        args.guidance_scale = 7.0
    print(args)
    # Prepare dataset-> same few shot training dataset
    # fix seed
    if args.seed != None: 
        print("fix seed")
        seed = args.seed 
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    
    imagenet_root = '/data/datasets/ImageNet'  
    args.imagenet_root = imagenet_root

    template = 'This is a photo of a {}'
    if args.testdata in ['ImageNet']:
        args.use_clip_official = True 

    test_dataset_name = [args.testdata] 
    args.savedir = os.path.join(args.savedir, f'{args.num_shot}shot', args.testdata)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    test_dataset = load_val_datasets(args, test_dataset_name)[0]   
    text_list = get_text_prompts_val([test_dataset], test_dataset_name, template=template, use_clip_official=args.use_clip_official)

    if hasattr(test_dataset, 'categories'):
        class_names = test_dataset.categories # for caltech256
    elif hasattr(test_dataset, '_classes'):
        class_names = test_dataset._classes
    else: 
        class_names = test_dataset.classes

    stage1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0").to('cuda')
    stage2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None).to('cuda')
    text_list = text_list[0]
    
    #generator = torch.manual_seed(1)
    for idx, c in enumerate(tqdm(class_names)):
        if idx < args.start:
            continue 
        c_dir =  os.path.join(args.savedir, c)
        print(f'classname {c}, saved in {c_dir}')
        if not os.path.exists(c_dir):
            os.makedirs(c_dir)
        text_prompt = text_list[idx]
        prompt_embeds, negative_embeds = stage1.encode_prompt(text_prompt)
        image = stage1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, num_images_per_prompt=args.num_shot, 
                       guidance_scale=args.guidance_scale, output_type="pt").images # 7.0 (1shot), 3.0 (more shot)
        prompt_embeds = prompt_embeds.repeat(args.num_shot,  1, 1)
        negative_embeds = negative_embeds.repeat(args.num_shot, 1, 1)
        #print(image.shape)
        image = stage2(
                 image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt").images #4.0
        image = pt_to_pil(image)
        for i in range(args.num_shot):
            image[i].save(os.path.join(c_dir, f'image_{i}.jpg'))
        


if __name__ == '__main__':
    main()