from .core import Smooth
import datetime
from time import time
import os
import numpy as np
import torch.nn as nn


def certify(base_classifier, num_classes, test_dataset, args, t=-1):

    print("\n-------- Certifying on the test set. --------")
    # prepare output file
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir) 
    if args.start == 0:
        f = open(args.outfile, 'w')
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    else:
        f = open(args.outfile, 'a') # 이어쓰기 
           
    base_classifier.eval()
    base_classifier = nn.DataParallel(base_classifier).cuda()

    # create the smooothed classifier g
    if t != -1:
        base_classifier.module.compute_attack = False # off gradient computation for memory 
        smoothed_classifier = Smooth(base_classifier, num_classes, args.sigma, t)
    else:
        smoothed_classifier = Smooth(base_classifier, num_classes, args.sigma)


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

        (x, label) = test_dataset[i]
        before_time = time()

        # certify the prediction of g around x 
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.c_batch)
        after_time = time()
        correct = int(prediction == label)

        print(f'certified process excuting...:{i}/{len(test_dataset)}')
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()