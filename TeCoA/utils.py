import shutil
import os
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR100, CIFAR10, StanfordCars, Food101, SUN397, EuroSAT, Caltech101,\
    Caltech256, Country211, Flowers102, PCAM, FGVCAircraft, OxfordIIITPet, STL10, DTD
import datetime
import time
from tqdm import tqdm
import clip
from torch.cuda.amp import autocast
import sys
sys.path.append('./TeCoA')
from cross_datasets import ISIC2018, ChestX, CropDisease
from attacks import attack_CW, attack_pgd, attack_auto
from models.model import multiGPU_CLIP, clip_img_preprocessing


best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ').replace('/', ' ')
    return class_names

def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.model_folder, filename)
    bestfile = os.path.join(args.model_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def load_imagenet_folder2name(path):
    dict_imagenet_folder2name = {}
    with open(path) as f:
        line = f.readline()
        while line:
            split_name = line.strip().split()
            if len(split_name) > 3: # for clip official name 
                cat_list =[]
                for i in range(len(split_name)-2):
                    cat_list.append(split_name[i+2])
                cat_name = " ".join(cat_list)
            else:
                cat_name = split_name[2]
            id = split_name[0]
            dict_imagenet_folder2name[id] = cat_name
            line = f.readline()
    # print(dict_imagenet_folder2name)
    return dict_imagenet_folder2name

cifar_preprocess = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

preprocess = transforms.Compose([
    transforms.ToTensor()
])

preprocess224 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

preprocess336 = transforms.Compose([
    transforms.Resize(336),
    transforms.CenterCrop(336),
    transforms.ToTensor()
])

preprocess224_interpolate = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def load_train_dataset(args):
    if args.dataset == 'cifar100':
        return CIFAR100(args.root, transform=preprocess, download=True, train=True)
    elif args.dataset == 'cifar10':
        #return CIFAR10(args.root, transform=preprocess, download=True, train=True)
        return CIFAR10(args.root, transform=cifar_preprocess, download=True, train=True)
    elif args.dataset in ['ImageNet', 'tiny-imagenet']:
        assert args.imagenet_root is not None
        print(f"Loading {args.dataset} from {args.imagenet_root}")
        if args.dataset == 'ImageNet':
            transform = preprocess224
        else:
            transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])  
        return ImageFolder(os.path.join(args.imagenet_root, 'train'), transform=transform)
    else:
        print(f"Train dataset {args.dataset} not implemented")
        raise NotImplementedError

def load_val_datasets(args, val_dataset_names, clip_preprocess=None):
    val_dataset_list = []
    for each in val_dataset_names:
        if each == 'cifar10':
            val_dataset_list.append(CIFAR10(args.root, transform=preprocess,
                               download=True, train=False)) # preprocess -> preproces224로 수정
        elif each == 'cifar100':
            val_dataset_list.append(CIFAR100(args.root, transform=preprocess,
                                            download=True, train=False))                                                         
        elif each == 'Caltech101':
            val_dataset_list.append(Caltech101(args.root, target_type='category', transform=preprocess224,
                                             download=True))
        elif each == 'PCAM':
            val_dataset_list.append(PCAM(args.root, split='test', transform=preprocess224,
                                             download=True))
        elif each == 'STL10':
            val_dataset_list.append(STL10(args.root, split='test',
                                               transform=preprocess, download=True))
        elif each == 'SUN397':
            val_dataset_list.append(SUN397(args.root,
                                               transform=preprocess224, download=True))
        elif each == 'StanfordCars':
            val_dataset_list.append(StanfordCars(args.root, split='test',
                                           transform=preprocess224, download=False))
        elif each == 'Food101':
            val_dataset_list.append(Food101(args.root, split='test',
                                           transform=preprocess224, download=True))
        elif each == 'oxfordpet':
            val_dataset_list.append(OxfordIIITPet(args.root, split='test',
                                           transform=preprocess224, download=True))
        elif each == 'EuroSAT':
            val_dataset_list.append(EuroSAT(args.root,
                                           transform=preprocess224, download=True))
        elif each == 'Caltech256':
            val_dataset_list.append(Caltech256(args.root, transform=preprocess224,
                                             download=True))
        elif each == 'flowers102':
            val_dataset_list.append(Flowers102(args.root, split='test',
                                           transform=preprocess224, download=True))
        elif each == 'Country211':
            val_dataset_list.append(Country211(args.root, split='test',
                                           transform=preprocess224, download=True))
        elif each == 'dtd':
            val_dataset_list.append(DTD(args.root, split='test',
                                           transform=preprocess224, download=True))
        elif each == 'fgvc_aircraft':
            val_dataset_list.append(FGVCAircraft(args.root, split='test',
                                           transform=preprocess224, download=True))
        elif each == 'ImageNet':
            if clip_preprocess != None:
                val_dataset_list.append(ImageFolder(os.path.join(args.imagenet_root, 'val'), transform=clip_preprocess))
            else:
                val_dataset_list.append(ImageFolder(os.path.join(args.imagenet_root, 'val'), transform=preprocess224))

        elif each == 'tiny-imagenet':
            val_dataset_list.append(ImageFolder(os.path.join(args.imagenet_root, 'val'), transform=preprocess))
        # cross-domain datasets 
        elif each == 'isic':
            val_dataset_list.append(ISIC2018(args.root, transform=preprocess224))
        elif each == 'chestx':
            val_dataset_list.append(ChestX(args.root, transform=preprocess224)) 
        elif each == 'cropdisease':
            val_dataset_list.append(CropDisease(args.root, transform=preprocess224))
        else:
            print(f"Val dataset {each} not implemented")
            raise NotImplementedError
        '''elif each == 'hateful_memes':
            val_dataset_list.append(HatefulMemes(args.root, splits=['test_seen', 'test_unseen'],
                                           transform=preprocess224_interpolate))'''
    return val_dataset_list

def get_text_prompts_train(args, train_dataset, template='This is a photo of a {}', use_clip_official=False):
    class_names = train_dataset.classes
    if args.dataset == 'ImageNet':
        if use_clip_official:
            filename = 'imagenet_classes_names_clip.txt'
        else:
            filename = 'imagenet_classes_names.txt'
        folder2name = load_imagenet_folder2name(filename)
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names
    if (use_clip_official == False) or (args.dataset != 'ImageNet'):
        class_names = refine_classname(class_names)
    texts_train = [template.format(label) for label in class_names]
    return texts_train

def get_text_prompts_val(val_dataset_list, val_dataset_name, template='This is a photo of a {}', use_clip_official=False):
    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        if hasattr(each, 'clip_prompts'):
            texts_tmp = each.clip_prompts 
        else:
            if hasattr(each, 'clip_categories'): # for caltech
                class_names = each.clip_categories
            else: 
                class_names = each.classes
                
            if val_dataset_name[cnt] in ['ImageNet', 'tiny-imagenet']:
                if use_clip_official:
                    filename = 'imagenet_classes_names_clip.txt'
                else:
                    filename = 'imagenet_classes_names.txt'
                folder2name = load_imagenet_folder2name(os.path.join('/data/jongheon_jeong/daewon/certified/TeCoA', filename))
                new_class_names = []
                for class_name in class_names:
                    new_class_names.append(folder2name[class_name])
                class_names = new_class_names
                #print("pass")

            if (use_clip_official == False) or (val_dataset_name[cnt] != 'ImageNet'):
                class_names = refine_classname(class_names)
            texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)
    assert len(texts_list) == len(val_dataset_list)
    return texts_list

def train(train_loader, texts, model, model_text, model_image, prompter, add_prompter,
          optimizer, scheduler, criterion, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.module.visual.train()

    num_batches_per_epoch = len(train_loader)

    alpha = args.train_stepsize
    attack_iters = args.train_numsteps

    # print('text token', texts)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        BATCH_SIZE = images.size(0)
        # print('bs', BATCH_SIZE)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        # print(images.min(), images.max())

        # with automatic mixed precision
        with autocast():
            if not args.VPbaseline:
                delta = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, images,
                                   target, text_tokens, alpha, attack_iters, 'l_2', epsilon=args.train_eps)
                # print('delta', delta.min(), delta.max())

                tmp = clip_img_preprocessing(images + delta)
            else:
                tmp = clip_img_preprocessing(images)

            prompted_images = prompter(tmp)
            prompt_token = None

            # for multiple GPU
            output, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

            loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.debug:
                break
            # break

            # if args.use_wandb:
            #     wandb.log({
            #         'training_loss': losses.avg,
            #         'training_acc': top1.avg
            #          })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'add_prompter': add_prompter.state_dict(),
                'vision_encoder_state_dict': model.module.visual.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg

def validate(val_dataset, dataset_num, val_dataset_name, texts_list, model, model_text, model_image,
             prompter, add_prompter, criterion, args):
    assert dataset_num == 1
    test_stepsize = args.test_stepsize

    for cnt in range(dataset_num):

        #val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        binary = ['PCAM', 'hateful_memes']
        attacks_to_run=['apgd-ce', 'apgd-dlr']
        if dataset_name in binary:
            attacks_to_run=['apgd-ce']

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()
        model.eval()

        # print(val_dataset_name, 'text token', texts_list)
        print("\n-------- evaluating on the test set. --------")
        # prepare output file
        if args.save_eval:
            outdir = os.path.dirname(args.eval_outfile)
            if not os.path.exists(outdir):
                os.makedirs(outdir) 
            if args.start==0:
                f = open(args.eval_outfile, 'w')
                print("idx\tlabel\tpredict_c\tpredict_adv\tcorrect_c\tcorrect_adv\ttime", file=f, flush=True)
            else:
                f = open(args.eval_outfile, 'a') # 이어쓰기 
        
        for i in range(len(val_dataset)):
            if i != 29524:
                continue
            if (i % args.skip != 0) or (i < args.start):
               continue
            if i == args.max:
               break
            if 'cifar' not in val_dataset_name:
                if i % 20 != 0 and not args.evaluate:
                    continue

            (images, target) = val_dataset[i] 
            images = images.unsqueeze(0)
            target = torch.Tensor([target]).long()

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)
            before_time = time.time()

            with autocast():

                # clean images, with prompt and without prompt
                # compute output
                with torch.no_grad():
                    prompt_token = None
                    output_prompt, _ = multiGPU_CLIP(model_image, model_text, model,
                                                     prompter(clip_img_preprocessing(images)), text_tokens,
                                                     prompt_token)

                torch.cuda.empty_cache()

                # generate adv example
                if args.CW:
                    delta_prompt = attack_CW(prompter, model, model_text, model_image, add_prompter, criterion,
                                             images, target, text_tokens,
                                             test_stepsize, args.test_numsteps, 'l_2', epsilon=args.test_eps)
                    attacked_images = images + delta_prompt
                elif args.autoattack:
                    attacked_images = attack_auto(model, images, target, text_tokens,
                                                  None, None, epsilon=args.test_eps, attacks_to_run=attacks_to_run)
                else:
                    #print(target)
                    delta_prompt = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion,
                                              images, target, text_tokens,
                                              test_stepsize, args.test_numsteps, 'l_2', epsilon=args.test_eps)
                    attacked_images = images + delta_prompt
                    #import pdb 
                    #pdb.set_trace()

                # compute output
                torch.cuda.empty_cache()
                with torch.no_grad():
                    prompt_token = add_prompter()
                    output_prompt_adv, _ = multiGPU_CLIP(model_image, model_text, model,
                                                         prompter(clip_img_preprocessing(attacked_images)),
                                                         text_tokens, prompt_token)
                # bl attack
                torch.cuda.empty_cache()

            after_time = time.time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print(f'evaluating process excuting...:{i}/{len(val_dataset)}')
            if args.save_eval:
                prediction_c = output_prompt.argmax(1)
                prediction_adv = output_prompt_adv.argmax(1)
                correct_c = int(prediction_c == target)
                correct_adv = int(prediction_adv == target)
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        i, target.cpu(), prediction_c.cpu().item(), prediction_adv.cpu().item(), 
                        correct_c, correct_adv, time_elapsed), file=f, flush=True)

        import pdb
        pdb.set_trace()
        torch.cuda.empty_cache()


    return 
