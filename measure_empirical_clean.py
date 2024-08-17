import os
import random
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn

import clip
from utils import *
import sys
from classifiers.clip_fewshot_model import CLIP_ZeroShot
from classifiers.resnet import resnet50
from TeCoA.utils import load_val_datasets, get_text_prompts_val, convert_models_to_fp32



def get_arguments():
    parser = argparse.ArgumentParser('Compute empirical clean accuracy of certified defense')
    
    parser.add_argument('--seed', type=int, default=0)

    # dataset 
    parser.add_argument('--root', type=str, default='./datasets/DATA',
                        help='dataset')
    parser.add_argument('--testdata', type=str, choices=['STL10', 'SUN397','StanfordCars', 'Food101',
                                                         'oxfordpet', 'Caltech256', 'flowers102',
                                                         'dtd','ImageNet','isic', 'EuroSAT', 'cropdisease'], help='test datasetname for certifying')
    parser.add_argument('--classifier_method', type=str, choices=['clip', 'resnet', 'resnet_RS']) # resnet_RS is gaussian-trained resnet: https://github.com/locuslab/smoothing
    parser.add_argument('--use_clip_official', type=bool, default=True, help='whether use clip official imagenet classname')

    parser.add_argument('--classifier_ckpt', type=str, default='')
    parser.add_argument('--result_file', type=str, default='', help='path for saving result log file')
    
    args = parser.parse_args()

    return args



def main(args):
    num_gpus = torch.cuda.device_count()
    args.num_workers = 0 

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

    print("Preparing test dataset.")
    template = 'This is a photo of a {}'
    test_dataset_name = args.testdata # dataset 
    test_dataset = load_val_datasets(args, [test_dataset_name])[0]          
    text_list = get_text_prompts_val([test_dataset], [test_dataset_name], template=template, use_clip_official=args.use_clip_official)
    
    if args.classifier_method == 'clip':
        print("use CLIP")
        if args.classifier_ckpt:
            print("use classifier checkpoint!")
            clip_model, _ = clip.load('ViT-B/32', jit=False) 
            convert_models_to_fp32(clip_model) # must!!
            classifier_ckpt = torch.load(args.classifier_ckpt)
            clip_model.load_state_dict(classifier_ckpt['model_state_dict']) # if not worked -> state_dict as key
        else:
            print("not use classifier checkpoint!")
            clip_model, _ = clip.load('ViT-B/32', jit=False) 
            convert_models_to_fp32(clip_model) # must!!
        classifier = CLIP_ZeroShot(clip_model, text_list=text_list)
        classifier.cuda()

    elif args.classifier_method in ['resnet', 'resnet_RS']:
        print("use ResNet")
        if (args.classifier_ckpt) and (args.classifier_method == 'resnet_RS'):
            classifier = resnet50(pretrained=False, use_ddp=True) 
            print("use resnet_RS and classifier checkpoint!")
            classifier_ckpt = torch.load(args.classifier_ckpt)
            classifier.load_state_dict(classifier_ckpt['state_dict'])
            classifier = nn.Sequential(classifier[0], classifier[1].module)
        elif args.classifier_ckpt:
            print("use classifier checkpoint!")
            classifier = resnet50(pretrained=False, use_ddp=False)
            classifier_ckpt = torch.load(args.classifier_ckpt)
            classifier.load_state_dict(classifier_ckpt['state_dict'])
        else:
            classifier = resnet50() 
            print("not use classifier checkpoint!")
    else:
        raise NotImplementedError("check --classifier_method args!")
    
    classifier.cuda().eval()

    result_df = pd.read_csv(args.result_file, delimiter='\t')
    correct = 0

    for i in tqdm(range(len(result_df))):
        #print(len(result_df))
        idx = result_df.loc[i, 'idx']
        #print(idx)
        predicted_label = result_df.loc[i, 'predict']
        is_correct = result_df.loc[i, 'correct']
        if is_correct:
            correct += 1
        else:
            if predicted_label != -1:
                continue
            else: #abstain
                (x, label) = test_dataset[idx]
                x = x.unsqueeze(0).to('cuda')
                logits_per_image = classifier(x)
                pred = logits_per_image.argmax(1) 
                correct += pred.cpu().eq(label)

    accuracy = (correct/len(result_df))*100
    print(f"empirical clean accuracy of this certified defense: {accuracy.item()}%")

if __name__ == '__main__':
    n = torch.cuda.device_count()  
    print(n)
    args = get_arguments()
    print(args)
    main(args) 