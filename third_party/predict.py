from .core import Smooth
import datetime
from time import time
import os
from tqdm import tqdm
import numpy as np
import torch
from third_party.attacks import attack_pgd, attack_auto
import torch.nn as nn
import torch.nn.functional as F


class SmoothingWrapper(torch.nn.Module):
    def __init__(self, model, noise, N=32, beta=1.0):
        super(SmoothingWrapper, self).__init__()
        self.model = model
        self.noise = noise
        self.N = N
        self.beta = beta

    def forward(self, x, t):
        xs = x.repeat((self.N, 1, 1, 1))
        logits = self.model(xs + self.noise, t, noise_add=False)
        avg_softmax = F.softmax(self.beta * logits, dim=1).mean(0, keepdim=True)
        logsoftmax = torch.log(avg_softmax.clamp(min=1e-20))
        return logsoftmax


def forward_with_timestep(images, model,  t):
    return model(images, t)


def predict(base_classifier, num_classes, test_dataset, args, t=-1):
    # create the smooothed classifier 
    print("\n-------- Empirical test on the test set. --------")
    # prepare output file
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    if (args.start == 0) or (args.validation_mode==True):
        f = open(args.outfile, 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    else:
        f = open(args.outfile, 'a') # 이어쓰기 
        # not update, only compute gradient 
    # not update, only compute gradient
    base_classifier.eval()
    for param in base_classifier.parameters():
            param.requires_grad_(False)
    base_classifier = nn.DataParallel(base_classifier).cuda()

    # iterate through the dataset
    for i in range(len(test_dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if args.validation_mode:
            if (i - args.start) % args.skip != 0:
               continue   
        else:
            if i % args.skip != 0:
               continue
        if i == args.max:
            break
        if i < args.start:
            continue
        
        #print(i)
        (x, label) = test_dataset[i] 

        #before_time = time()
        # certify the prediction of g around x
        x = x.unsqueeze(0).to('cuda')
        label = torch.Tensor([label]).long().to('cuda')

        if args.method in ['DiffusionRobustModel', 'IFRobustModel']:
            base_classifier.module.compute_attack = True

        if not args.random_noise_attack:
            noise = torch.randn((args.num_noise_vec, *x.shape[1:]), device=x.device) * args.sigma # noise is not change in attack iteration 
        else:
            noise = None 
        #print(noise, args.random_noise_attack)
        if args.attack_type == 'pgd':
            x = x.repeat(args.num_noise_vec, 1, 1, 1) # [num_noise_vec, 3, 224, 224]
            #base_classifier.module.compute_attack = True
            delta_prompt = attack_pgd(base_classifier, x, t, target=label, noise=noise, use_random_noise=args.random_noise_attack, num_noise_vec=args.num_noise_vec, epsilon=args.test_eps,
                                    test_step_size=args.test_stepsize, attack_iters=args.test_numsteps, norm=args.norm_type)
            attacked_images = x + delta_prompt.repeat(1, args.num_noise_vec, 1, 1).view_as(x)
            attacked_images = attacked_images[:1] # only one is okay!

        elif args.attack_type == 'auto':
            binary = ['PCAM', 'hateful_memes']
            attacks_to_run=['apgd-ce', 'apgd-dlr']
            if args.testdata in binary:
                attacks_to_run=['apgd-ce']
            smoothing_wrapper = SmoothingWrapper(base_classifier, noise=noise, N=args.num_noise_vec)
            smoothing_wrapper = smoothing_wrapper.eval()
            import functools
            forward_pass = functools.partial(
                forward_with_timestep, 
                model=smoothing_wrapper, 
                t=t
            )
            attacked_images = attack_auto(forward_pass, x, target=label, epsilon=args.test_eps, attacks_to_run=attacks_to_run)
        
        else: # just direct predict process
            attacked_images = x
            

        if t != -1:
            base_classifier.module.compute_attack = False # off gradient computation for memory 
            smoothed_classifier = Smooth(base_classifier, num_classes, args.sigma, t)
        else:
            smoothed_classifier = Smooth(base_classifier, num_classes, args.sigma)
        
        attacked_images = attacked_images.squeeze()
        label = label.item()
        before_time = time()
        prediction = smoothed_classifier.predict(attacked_images, n=args.N0, alpha=args.alpha, batch_size=args.c_batch)
        after_time = time()
        correct = int(prediction == label)
        # for diffusion robust model debugging, save images
        print(f'empirical test process excuting...:{i}/{len(test_dataset)}')
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        radius = args.test_eps
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()