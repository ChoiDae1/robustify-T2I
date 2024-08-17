import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn

import clip
import numpy as np
#from utils import *
from classifiers.clip_fewshot_model import CLIP_ZeroShot
from classifiers.resnet import resnet50
from third_party.certify import certify
from third_party.predict import predict
from diffusion_robust_model import DiffusionRobustModel
from IF_robust_model import IFRobustModel
from TeCoA.utils import load_val_datasets, get_text_prompts_val, convert_models_to_fp32
from utils import return_prompt



def get_arguments():
    parser = argparse.ArgumentParser('Certifying CLIP ZeroShot')
    
    parser.add_argument('--seed', type=int, default=0)
    # certifying parameters
    parser.add_argument('--start', type=int, default=0) 
    parser.add_argument('--skip', type=int, default=-1, help='you dont need to revise this param')
    parser.add_argument('--nrows', type=int, default=500) 
    parser.add_argument('--max', type=int, default=-1)
    parser.add_argument('--N0', type=int, default=100) #smooth sample size for predict
    parser.add_argument('--N', type=int, default=10000) # smooth sample size for cerity
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--sigma', type=float, default=0.25, help='noise level for certifying') # check
    parser.add_argument('--empirical', action='store_true') # for caculation empirical 
    parser.add_argument('--validation_mode', type=bool, default=False)

    #empirical
    parser.add_argument('--num_noise_vec', type=int, default=32)
    parser.add_argument('--norm_type', type=str, default='l_2', choices=['l_2', 'l_inf'])
    parser.add_argument('--test_eps', type=float, default=1.0) # check  empirical=True면!
    parser.add_argument('--test_numsteps', type=int, default=100)
    parser.add_argument('--random_noise_attack', type=bool, default=False) # default is false following smoothadv
    parser.add_argument('--attack_type', type=str, default='pgd', choices=['pgd', 'auto', 'clean'])

    # dataset 
    parser.add_argument('--root', type=str, default='./datasets/DATA',
                        help='dataset')
    parser.add_argument('--imagenet_root', type=str, default='/data/datasets/ImageNet', help='imagenet root directory')
    parser.add_argument('--testdata', type=str, choices=['STL10', 'SUN397','StanfordCars', 'Food101',
                                                         'oxfordpet', 'Caltech256', 'flowers102',
                                                         'dtd','ImageNet','isic', 'EuroSAT', 'cropdisease'], help='test datasetname for certifying')
    parser.add_argument('--use_clip_official', type=bool, default=True, help='whether use clip official imagenet classname')

    # model
    parser.add_argument('--method', type=str, choices=['RS', 'DiffusionRobustModel', 'IFRobustModel']) # RS: CLIP-Smooth in out paper
    parser.add_argument('--diffusion_ckpt', type=str, default='') # for DiffusionRobustModel
    parser.add_argument('--classifier_method', type=str, choices=['clip', 'resnet'])
    parser.add_argument('--classifier_ckpt', type=str, default='')

    # save
    parser.add_argument('--outfile', type=str, default='', help='path for saving result log file')
    args = parser.parse_args()

    return args

def main(args):
    num_gpus = torch.cuda.device_count()
    args.num_workers = 0 # 0으로 해야 error 발생 x

    # fix seed
    if args.seed != None: 
        print("fix seed")
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    print("Preparing test dataset.")

    template = 'This is a photo of a {}'
    test_dataset_name = args.testdata # dataset 하나씩만
    test_dataset = load_val_datasets(args, [test_dataset_name])[0]          
    text_list = get_text_prompts_val([test_dataset], [test_dataset_name], template=template, use_clip_official=args.use_clip_official)
    
    import math 
    if args.validation_mode:
        args.start = 1
        args.nrows = 200

    args.skip = math.floor(len(test_dataset)/args.nrows)
    print(f'skip factor is {args.skip}')

    num_classes = len(text_list[0])
    print(text_list[0])
    print(f'num classes : {num_classes}')
    print(f'sigma: {args.sigma}')

    # Certify 
    if args.method=='DiffusionRobustModel': 
        assert (args.testdata == 'ImageNet') and (args.diffusion_ckpt != None) # [Carlini et al, 2024] only ImageNet is supported and use diffusion checkpoint in their github
        base_classifier = DiffusionRobustModel(diffusion_ckpt=args.diffusion_ckpt, 
                                               classifier_method=args.classifier_method, classifier_ckpt=args.classifier_ckpt, text_list=text_list, 
                                               sigma=args.sigma, num_classes=num_classes)
        base_classifier = base_classifier.cuda().eval()
        # Get the timestep t corresponding to noise level sigma
        t = base_classifier.estimate_timestep()
        args.c_batch = 100 * num_gpus
        
        if args.empirical:
            args.test_stepsize = (args.test_eps/args.test_numsteps)*(4/3) 
            predict(base_classifier, num_classes, test_dataset, args, t)
        else:
            certify(base_classifier, num_classes, test_dataset, args, t)

    elif args.method=='IFRobustModel':
        base_classifier = IFRobustModel(lora_ckpt=args.diffusion_ckpt, prompt=args.prompt,
                                        classifier_method=args.classifier_method, classifier_ckpt=args.classifier_ckpt,
                                        text_list=text_list, sigma=args.sigma, num_classes=num_classes)
        args.c_batch = 60 * num_gpus
        t = base_classifier.estimate_timestep()
        print(f'sigma {args.sigma}-estimated timestep is {t}')
       
        if args.empirical:
            args.test_stepsize = (args.test_eps/args.test_numsteps)*(4/3) 
            predict(base_classifier, num_classes, test_dataset, args, t)
        else:
            certify(base_classifier, num_classes, test_dataset, args, t)
    
    elif args.method == 'RS':
        if args.classifier_method == 'zeroshot':
            print("use CLIP")
            if args.classifier_ckpt:
                print("use classifier checkpoint!")
                clip_model, _ = clip.load('ViT-B/32', jit=False) # clip github 참조
                convert_models_to_fp32(clip_model) # must!!
                classifier_ckpt = torch.load(args.classifier_ckpt)
                clip_model.load_state_dict(classifier_ckpt['model_state_dict'])
            else:
                print("not use classifier checkpoint!")
                clip_model, _ = clip.load('ViT-B/32', jit=False) # clip github 참조
                convert_models_to_fp32(clip_model) # must!!

            base_classifier = CLIP_ZeroShot(clip_model, text_list=text_list)
            base_classifier.cuda()

        elif args.classifier_method == 'resnet':
            print("use ResNet")
            if args.classifier_ckpt:
                base_classifier = resnet50(pretrained=False, use_ddp=True) 
                print("use classifier checkpoint!")
                classifier_ckpt = torch.load(args.classifier_ckpt)
                base_classifier.load_state_dict(classifier_ckpt['state_dict'])
                base_classifier = nn.Sequential(base_classifier[0], base_classifier[1].module)
            else:
                base_classifier = resnet50() 
                print("not use classifier checkpoint!")
        else:
            raise NotImplementedError("check --classifier args!")

        args.c_batch = 256 * num_gpus
            
        if args.empirical:
            args.test_stepsize = (args.test_eps/args.test_numsteps)*(4/3) 
            predict(base_classifier, num_classes, test_dataset, args)
        else:
            certify(base_classifier, num_classes, test_dataset, args)
    else:
        raise NotImplementedError("check --method args!")
        

if __name__ == '__main__':
    n = torch.cuda.device_count()  
    print(n)
    args = get_arguments()
    args.prompt = return_prompt(args.testdata, personalized=True)
    print(args)
    main(args) 