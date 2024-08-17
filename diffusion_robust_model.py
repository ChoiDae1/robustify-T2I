import torch
import torch.nn as nn 
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from torchvision.transforms.functional import to_pil_image
from classifiers.clip_fewshot_model import CLIP_ZeroShot
from classifiers.resnet import resnet50
from TeCoA.utils import convert_models_to_fp32
import clip
from abc import *
import os

class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False



class BaseDiffusionRobustModel(nn.Module):
    def __init__(self, classifier_method='', classifier_ckpt='',sigma=0.0, num_classes=-1, text_list=None):
        super().__init__()
        self.sigma = sigma
        self.num_classes = num_classes
        if classifier_method=='clip':
            print("use clip zeroshot")
            if classifier_ckpt:
                print("use classifier checkpoint!")
                clip_model, _ = clip.load('ViT-B/32', jit=False) # clip github 참조
                convert_models_to_fp32(clip_model) # must!!
                classifier_ckpt = torch.load(classifier_ckpt)
                clip_model.load_state_dict(classifier_ckpt['model_state_dict']) # if not worked -> state_dict as key
                self.classifier = CLIP_ZeroShot(clip_model, text_list=text_list)
            else:
                print("not use classifier checkpoint!")
                clip_model, _ = clip.load('ViT-B/32', jit=False) # clip github 참조
                convert_models_to_fp32(clip_model) # must!!
                self.classifier = CLIP_ZeroShot(clip_model, text_list=text_list)
        elif classifier_method=='resnet':
            print("use resnet")
            #self.classifier = ResNet18()
            #self.classifier = pre_resnet18(num_classes=200, stride=2)
            self.classifier = resnet50()
            if classifier_ckpt:
                print("use classifier checkpoint!")
                classifier_ckpt = torch.load(classifier_ckpt)
                self.classifier.load_state_dict(classifier_ckpt['model_state_dict']) # if not worked -> state_dict as key
            else:
                print("not use classifier checkpoint!")
        else: 
            print("not use classifier, this is debugging mode!")
      
    @abstractmethod
    def estimate_timestep(self):
        pass

    @abstractmethod
    def denoise(self):
        pass

    def save_image(self, x, t, savedir=None, savename=None):
        # save input
        x_pil = to_pil_image(x[0])
        x_pil.save(os.path.join(savedir,f'{savename}_input.png'))
        
        noise = torch.randn_like(x, device=f'cuda:{x.get_device()}') * 0.25 # 0.25 is noise
        x_noised = (x + noise) * 2 -1 # convert [-1~1]
        
        # save noised input
        x_noised_pil = to_pil_image((x_noised[0] / 2 + 0.5).clamp(0, 1))
        x_noised_pil.save(os.path.join(savedir, f'{savename}_noised.png'))
        
        x_noised = self.scaling_factor.item()*x_noised # scaling
        imgs = self.denoise(x_noised, t)
        #print(self.scaling_factor.item())
        # save denoised input 
        imgs_pil  = to_pil_image((imgs[0] / 2 + 0.5).clamp(0, 1))
        imgs_pil.save(os.path.join(savedir, f'{savename}_denoised.png'))



class DiffusionRobustModel(BaseDiffusionRobustModel):
    def __init__(self, diffusion_ckpt='', classifier_method='', classifier_ckpt='', sigma=0.0, num_classes=-1, text_list=None, compute_attack=False):
        super().__init__(classifier_method, classifier_ckpt, sigma, num_classes, text_list)

        if diffusion_ckpt:
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(Args(), model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(diffusion_ckpt)
            )
            model.eval().to('cuda')
            self.model = model 
            self.diffusion = diffusion 
        else:
            raise ValueError("Diffusion's checkpoint path must be to needed")
        self.compute_attack = compute_attack
        #self.model.to('cuda')
    
    def estimate_timestep(self):
        target_sigma = self.sigma * 2 
        real_sigma = 0
        t = 0
        while real_sigma < target_sigma:
            t += 1
            a = self.diffusion.sqrt_alphas_cumprod[t]
            b = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
            real_sigma = b / a  
        self.scaling_factor = self.diffusion.sqrt_alphas_cumprod[t]

        return t
    
    def forward(self, x, t, only_denoise=False, noise_add=True):
        if noise_add:
            noise = torch.randn_like(x) * self.sigma
            x_noised = x + noise # add noise [0~1] image space
        else:
            x_noised = x 
        x_noised = x_noised * 2 -1 # convert [-1~1]
        x_noised = self.scaling_factor.item()*x_noised # scaling
        imgs = self.denoise(x_noised, t, multistep=False)
        imgs = (imgs / 2 + 0.5).clamp(0, 1) # convert [0~1]
        if only_denoise:
            return imgs
        else:
            with torch.set_grad_enabled(self.compute_attack):
                out = self.classifier(imgs)

        return out
    
    def denoise(self, x_t_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_t_start)).to(x_t_start.device)

        with torch.set_grad_enabled(self.compute_attack):
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_t_start)).to(x_t_start.device)
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out



