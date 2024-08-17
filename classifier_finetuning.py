import os
import random
import argparse
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

import torch.backends.cudnn as cudnn
import numpy as np
from math import ceil
from torch.utils.data import DataLoader

import clip
from IF_robust_model import IFRobustModel
from diffusion_robust_model import DiffusionRobustModel
from TeCoA.utils import load_train_dataset, get_text_prompts_train, AverageMeter, ProgressMeter, \
                            convert_models_to_fp32, load_val_datasets, get_text_prompts_val
from TeCoA.models.model import clip_img_preprocessing
from utils import GeneratedDataset, cosine_lr
from classifiers.resnet import resnet50
from utils import return_prompt


def get_arguments():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_train_steps', type=int, default=None) # 10 at 1shot
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-7) #5e-7 at 1shot
    parser.add_argument('--min_lr', type=float, default=0.0) # for cosine scheduler 
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--warmup_length', type=int, default=0) # for few shot training 
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256) # 약 전체 dataset size의 1/4하면 될듯??
    parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
    parser.add_argument('--tuning_last', action='store_true')
    parser.add_argument('--classifier_method', type=str, choices=['clip', 'resnet'])
    parser.add_argument('--classifier_ckpt', type=str, default='', help='checkpoint path for CLIP')
    parser.add_argument('--use_clip_official', type=bool, default=True, help='use clip official imagenet classname')
    # dataset
    parser.add_argument('--root', type=str, default='./datasets/DATA', help='dataset')
    parser.add_argument('--imagenet_root', type=str, default='/data/datasets/ImageNet', help='dataset')
    parser.add_argument('--use_generated_dataset', action='store_true') # generated dataset use
    parser.add_argument('--generated_data_root', type=str, default='./datasets/generated_DATA/1shot')
    parser.add_argument('--num_shot', type=int, default=1)
    parser.add_argument('--testdata', type=str, default='ImageNet',
                        help='dataset for finetuning', choices=['STL10', 'SUN397','StanfordCars', 'Food101',
                                                         'oxfordpet', 'Caltech256', 'flowers102',
                                                         'dtd','ImageNet','isic', 'EuroSAT', 'cropdisease']) # 수정
    # for classifier fine-tuning, self-personalization ckpt is needed
    parser.add_argument('--diffusion_ckpt', type=str, default=None, required=True) 

    # save
    parser.add_argument('--out_dir', type=str, default='') # 수정
    parser.add_argument('--save_freq', type=int, default=8,
                        help='save frequency')
    args = parser.parse_args()

    return args


def main(args):
    print(args.use_generated_dataset)
    # fix seed
    if args.seed != None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    
    assert args.max_train_steps != -1 
    ngpus = torch.cuda.device_count()
    print(ngpus)
    args.num_workers = ngpus * 2
    args.generated_data_root = os.path.join(args.generated_data_root, args.testdata)
    
    # CLIP
    if args.classifier_method == 'clip':
        classifier, _ = clip.load('ViT-B/32', jit=False) # clip github 참조
        convert_models_to_fp32(classifier) # must!!
    elif args.classifier_method == 'resnet':
        classifier = resnet50()
        
    template = 'This is a photo of a {}'
    imagenet_root = '/data/datasets/ImageNet'  
    args.imagenet_root = imagenet_root 

    train_sampler = None
    train_dataset = GeneratedDataset(args)
    dataset_for_texts = load_val_datasets(args, [args.testdata])[0]
    texts_train = get_text_prompts_val([dataset_for_texts], [args.testdata], template=template, 
                                        use_clip_official=args.use_clip_official)[0]
    class_names = train_dataset.class_names
    
    num_classes = len(class_names)
    if args.batch_size > num_classes:
       args.batch_size = num_classes
    
    print(texts_train)
    print(f'batchsize is {args.batch_size}')

    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=True, sampler=train_sampler)

    devices = list(range(ngpus))
    classifier = torch.nn.DataParallel(classifier, device_ids=devices).cuda() # for data parallel

    if (args.tuning_last) and (args.classifier_method=='resnet'):
        for param in classifier.module[1].parameters():
            param.requires_grad = False
        for param in classifier.module[1].linear.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(classifier.module[1].linear.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(classifier.module.parameters(), lr=args.lr, weight_decay=args.wd)
    
    denoiser = IFRobustModel(lora_ckpt=args.diffusion_ckpt, prompt=args.prompt)
    denoiser = denoiser.cuda()
    
    if args.max_train_steps != None:
       args.num_train_epochs = math.ceil(args.max_train_steps / len(train_loader))
    else:
       args.max_train_steps = args.num_train_epochs * len(train_loader)

    # set scheduler 
    if args.use_scheduler:
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                            args.max_train_steps, args.min_lr)
        
    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    # train 
    for epoch in range(args.start_epoch, args.num_train_epochs):
        epoch_stats = {}
        epoch_stats['step'] = global_step
        classifier.module.train()
        losses = AverageMeter('Loss', ':.4e')

        for i, (images, target) in enumerate(train_loader):
            if args.use_scheduler:
                if epoch != -1:
                    scheduler(global_step)

            optimizer.zero_grad()
            batch_size = images.size(0)
            
            images = images.to('cuda')
            text_tokens = clip.tokenize(texts_train).to('cuda') # clean_mao, mao not need to target preprocessing

            # compute denoising 
            outer_batch_sz = images.shape[0]
            inner_batch_sz = 60 * ngpus
            denoised_images = torch.zeros_like(images).to('cuda')
            start_idx = 0 
            for i in range(ceil(outer_batch_sz / inner_batch_sz)):
                cur_batch_size = min(outer_batch_sz, inner_batch_sz)
                cur_batch_image = images[start_idx:start_idx+cur_batch_size, :, :]

                cur_batch_image = 2 * cur_batch_image - 1 #[-1, 1]
                # random noise sampling
                timesteps = torch.randint(
                    0, denoiser.scheduler.config.num_train_timesteps, (cur_batch_size,), device=images.device
                )
                timesteps = timesteps.long()
                noise = torch.randn_like(cur_batch_image)
                noisy_model_input = denoiser.scheduler.add_noise(cur_batch_image, noise, timesteps)
                denoised_list = []
                for idx, timestep in enumerate(timesteps):
                    timestep = int(timestep.item())
                    temp_denoised = denoiser.denoise(noisy_model_input[idx].unsqueeze(0), [timestep]) # [0, 1], scheduler.step only timestep as integer
                    temp_denoised = (temp_denoised / 2 + 0.5).clamp(0, 1) # convert [0~1]
                    denoised_list.append(temp_denoised)
                inner_denoised_images = torch.cat(denoised_list, dim=0)
                denoised_images[start_idx:start_idx+cur_batch_size, :, :] = inner_denoised_images
                outer_batch_sz -= cur_batch_size
                start_idx += cur_batch_size     

            denoised_images = clip_img_preprocessing(denoised_images) # [-1~1]
            image_features, text_features = classifier(denoised_images, text_tokens, return_features=True) #clip 코드 수정 필요
            
            if args.classifier_method == 'clip':
                logits_per_image = image_features @ text_features.T
                target = target.to('cuda')
                contrastive_loss = F.cross_entropy(logits_per_image, target)
                total_loss = contrastive_loss 
            elif args.classifier_method == 'resnet':
                logits_per_image = classifier(denoised_images)
                target = target.to('cuda')
                total_loss = F.cross_entropy(logits_per_image, target)

            total_loss.backward()
            losses.update(total_loss.item(), batch_size)
            optimizer.step()
            progress_bar.update(1)
            global_step += 1

        train_loss = losses.avg
        print(f'Epoch {epoch} training loss: {train_loss}')
        if (epoch+1)==args.num_train_epochs:
            if args.out_dir is not None:
                os.makedirs(args.out_dir, exist_ok=True)
                save_path = os.path.join(args.out_dir, f'checkpoint-last.pt')
                torch.save({
                            'epoch':epoch,
                            'model_state_dict':classifier.module.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'loss': train_loss
                            }, save_path)
                print(f'saving is completed at {save_path}')

if __name__ == '__main__':
    args = get_arguments()
    args.prompt = return_prompt(args.testdata, personalized=True)
    main(args)