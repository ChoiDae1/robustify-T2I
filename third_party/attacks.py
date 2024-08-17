import torch
import sys
#sys.path.append('../TeCoA')
from TeCoA.models.model import *
import torch.nn.functional as F
from autoattack import AutoAttack


lower_limit, upper_limit = 0, 1
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, t, target, noise, use_random_noise=False,  num_noise_vec=32,
               epsilon=1.0, test_step_size=-1, attack_iters=100, norm='l_2', smoothing=True): # x shape must be [channel, h, w]
    #print(target)
    delta = torch.zeros((len(target), *X.shape[1:])).cuda() # [1, 3, 224, 224], image 마다 delta 달라짐.  
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        #d_flat = delta.view(-1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    
    if (not smoothing) and (num_noise_vec != 1):
        num_noise_vec = 1 
        
    delta.data.add_(X[::num_noise_vec])
    delta.data.clamp_(lower_limit, upper_limit).sub_(X[::num_noise_vec])
    delta.requires_grad = True

    for _ in range(attack_iters):
        if smoothing:
            adv = X + delta.repeat(1, num_noise_vec, 1, 1).view_as(X)
            if not use_random_noise:
                adv = adv + noise
        else:
            adv = X + delta.view_as(X)
        
        if t != -1:
           logits = model(adv, t, noise_add=use_random_noise) #  output: [1xnum_noise_vec, num_classes]
        else: # for method general classifier
           logits = model(adv)
        
        if smoothing:
            softmax = F.softmax(logits, dim=1)
            # average the prob across noise
            average_softmax = softmax.reshape(-1, num_noise_vec, logits.shape[-1]).mean(1, keepdim=True).squeeze(1) #[1, num_classes]
            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            loss = F.nll_loss(logsoftmax, target)
        else: # not smoothing method like Vanilla-CLIP, Mao
            loss = F.cross_entropy(logits, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + test_step_size * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * test_step_size).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d.data.add_(x[::num_noise_vec])
        d.data.clamp_(lower_limit, upper_limit).sub_(x[::num_noise_vec])
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_auto(forward_pass, images, target,
                         attacks_to_run=['apgd-ce', 'apgd-dlr'], epsilon=0):
    adversary = AutoAttack(forward_pass, norm='L2', eps=epsilon, version='rand', verbose=False)
    adversary.attacks_to_run = attacks_to_run
    x_adv = adversary.run_standard_evaluation(images, target, bs=images.shape[0])

    return x_adv