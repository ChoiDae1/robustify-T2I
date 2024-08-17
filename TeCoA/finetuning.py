from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import random
import warnings
# import wandb

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import clip
from models import prompters
from models.prompters import TokenPrompter, NullPrompter
from utils import save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, get_text_prompts_train, get_text_prompts_val
from utils import load_train_dataset, load_val_datasets
from utils import train, validate



def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=1,
                        help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epoch5s')
    parser.add_argument("--mix_alpha", type=float, default=-1,
                        help="interpolation")

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-7,  ## Why so large
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_eps', type=float, default=1.0,
                        help='momentum') # 수정
    parser.add_argument('--train_numsteps', type=int, default=2)
    parser.add_argument('--train_stepsize', type=int, default=1.0) # 수정
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--imagenet_root', type=str, default=None)
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='null_patch',
                        choices=['null_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=0,
                        help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='/data/jongheon_jeong/daewon/certified/datasets/DATA',
                        help='dataset') # 수정
    parser.add_argument('--dataset', type=str, default='ImageNet',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--VPbaseline', action='store_true')
    parser.add_argument('--CW', action='store_true')
    parser.add_argument('--autoattack', action='store_true')
    parser.add_argument('--train_class_count', type=int, default=90)
    parser.add_argument('--last_num_ft', type=int, default=-1)

    parser.add_argument('--noimginprop', action='store_true')
    
    # evaluate
    parser.add_argument('--evaluate', 
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--valdata', type=str, choices=['cifar10', 'cifar100', 'Caltech101',
                                                         'PCAM', 'STL10', 'SUN397','StanfordCars', 'Food101',
                                                         'oxfordpet', 'EuroSAT', 'Caltech256', 'flowers102',
                                                         'Country211', 'dtd', 'fgvc_aircraft',  'ImageNet','tiny-imagenet', 
                                                         'isic', 'chestx', 'cropdisease'], help='test datasetname for certifying')
    parser.add_argument('--test_eps', type=float, default=1.0,
                        help='momentum') # check
    parser.add_argument('--test_numsteps', type=int, default=100) # 상관없음t
    #parser.add_argument('--test_stepsize', type=int, default=0.013)
    #parser.add_argument('--resume', type=str, default='./save/models/perturbation_bound_1.0/checkpoint.pth.tar',
    #                    help='path to resume from checkpoint') # check
    parser.add_argument('--resume', type=str, default='',
                        help='path to resume from checkpoint') # check 없으면 vanila clip임 
    parser.add_argument('--skip', default=50, type=int) # 밑에서 수정됨! 
    parser.add_argument('--start', default=0)
    parser.add_argument('--max', default=-1, type=int)
    parser.add_argument('--save_eval', default=True, type=bool)
    parser.add_argument('--eval_outfile', default='../output/jh/food_emp_1.0_pgd.txt') # check
    args = parser.parse_args()

    '''args.filename = '{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}_addp_{}'. \
        format(args.name, args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial,
               args.add_prompt_size)'''

    return args

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    global best_acc1, device
    
    #args.train_eps = args.train_eps / 255.
    #args.test_eps = args.test_eps / 255.
    #args.train_stepsize = args.train_stepsize / 255.
    args.test_stepsize = (args.test_eps/args.test_numsteps)*(4/3) 
    args.filename = 'train_perturbation_bound_{}_train_stepsize_{}'.format(args.train_eps, args.train_stepsize)
    n = torch.cuda.device_count()
    args.num_workers = n * 2
    print(args)
    
    # fix seed
    if args.seed != None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

    imagenet_root = '/data/datasets/ImageNet'
    args.imagenet_root = imagenet_root

    #if args.imagenet_root is not None:
    #    imagenet_root = args.imagenet_root

    add_prompt_len = 0

    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
    model_text, model_image = None, None

    convert_models_to_fp32(model)
    model = torch.nn.DataParallel(model)  # .to(device)
    model.eval()

    prompter = NullPrompter()  # .to(device)
    add_prompter = TokenPrompter(add_prompt_len)  # .to(device)

    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    # define criterion and optimizer
    # we finetune the image module parameters only
    if args.last_num_ft == -1:
        optimizer = torch.optim.SGD(model.module.visual.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(list(model.module.visual.parameters())[-args.last_num_ft:],
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    args.start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            print(args.start_epoch)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            if args.mix_alpha > 0:
                alpha = args.mix_alpha
                checkpoint_ori = torch.load('original_clip.pth.tar')
                theta_ori = checkpoint_ori['vision_encoder_state_dict']
                theta_rob = checkpoint['vision_encoder_state_dict']

                theta = {
                    key: (1 - alpha) * theta_ori[key] + alpha * theta_rob[key]
                    for key in theta_ori.keys()
                }
                model.module.visual.load_state_dict(theta)

            else:
                model.module.visual.load_state_dict(checkpoint['vision_encoder_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')
    
    # for train
    if args.evaluate==False:
        train_sampler = None
        train_dataset = load_train_dataset(args)
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size, pin_memory=True,
                                num_workers=args.num_workers, shuffle=True, sampler=train_sampler)
        texts_train = get_text_prompts_train(args, train_dataset, template=template)
        scaler = GradScaler()
        total_steps = len(train_loader) * args.epochs
        scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)
        # make dir
        refined_template = template.lower().replace(' ', '_')
        args.filename = f'{args.filename}_template_{refined_template}'

        args.model_folder = os.path.join(args.model_dir, args.filename)
        if not os.path.isdir(args.model_folder):
            os.makedirs(args.model_folder)

    val_dataset_name = [args.valdata]
    val_dataset_list = load_val_datasets(args, val_dataset_name)
    print(len(val_dataset_list[0]))
    #
    import math 
    args.skip = math.floor(len(val_dataset_list[0])/500)
    print(f'skip factor is {args.skip}')
    val_sampler = None
    val_loader_list = [DataLoader(each,
                                  batch_size=1, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=val_sampler) for each in
                       val_dataset_list] # batch_size 1로 할경우, export CUDA_VISIBLE_DEVICES=0해야함. 
    texts_list = get_text_prompts_val(val_dataset_list, val_dataset_name, template=template)
    print(texts_list)
    cudnn.benchmark = True

    # wandb
    # if args.use_wandb:
    #     wandb.init(project='Visual Prompting')
    #     wandb.config.update(args)
    #     wandb.run.name = args.filename
    #     wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1_mean = validate(val_dataset_list[0], len(val_dataset_list), val_dataset_name, texts_list, model, model_text, model_image,
                             prompter, add_prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, texts_train, model, model_text, model_image, prompter, add_prompter, optimizer, scheduler,
              criterion, scaler, epoch, args)

        # evaluate on validation set
        '''if epoch % args.validate_freq == 0:
            acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                                 prompter, add_prompter, criterion, args)'''

        # remember best acc@1 and save checkpoint
        #is_best = acc1_mean > best_acc1
        #best_acc1 = max(acc1_mean, best_acc1)
        is_best = False
        best_acc1 = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'vision_encoder_state_dict': model.module.visual.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    # wandb.run.finish()


if __name__ == '__main__':
    n = torch.cuda.device_count()
    print(n)
    args = parse_option()
    # fix seed
    main(args)
