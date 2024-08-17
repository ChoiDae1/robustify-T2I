import torch
from diffusers import DiffusionPipeline
from torchvision.transforms.functional import to_pil_image
from diffusion_robust_model import BaseDiffusionRobustModel
import os 
import torch.nn.functional as F


class IFRobustModel(BaseDiffusionRobustModel): 
    def __init__(self, lora_ckpt='',  prompt='', guidance_scale=0.0,
                 classifier_method='', classifier_ckpt='', sigma=-1, num_classes=-1, text_list=None
                 ,compute_attack=False
                 ):
        super().__init__(classifier_method, classifier_ckpt, sigma, num_classes, text_list)
        self.sigma = sigma
        self.generator = None # default setting or None
        self.compute_attack = compute_attack
        if guidance_scale > 1.0:
            self.guidance_scale = guidance_scale
            self.do_classifier_free_guidance = True
        else:
            self.do_classifier_free_guidance = False
        self.set_pipeline_component(lora_ckpt, prompt=prompt)

    def set_pipeline_component(self, lora_ckpt='', prompt=''):
        print("use super-resolution pipeline")
        pipeline = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0")
        if lora_ckpt:
            print(f"lora checkpoint loaded...=> {lora_ckpt}")
            pipeline.load_lora_weights(lora_ckpt) # load lora model 
        else:
            print("not use lora checkpoint")

        # pre-computing text embedding 
        self.prompt_embeds, self.negative_prompt_embeds = pipeline.encode_prompt(prompt, do_classifier_free_guidance=self.do_classifier_free_guidance) # we dont use classifier guidance
        # unet, scheduler load
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler # 얘는 replica가 복제x, 공유함. 
        self.scheduler = self.scheduler.__class__.from_config(self.scheduler.config, variance_type="fixed_small") 
        del pipeline
        
    def estimate_timestep(self, control_factor=1): 
        assert self.sigma != -1
        target_sigma_list = []
        t_list = []
        self.scaling_factor_list = []

        target_sigma = self.sigma * 2
        target_sigma_list.append(target_sigma)

        for target_sigma in target_sigma_list: 
            real_sigma = 0
            t = 0
            while real_sigma < target_sigma:
                t += 1
                alphas_cumprod = self.scheduler.alphas_cumprod
                a = alphas_cumprod[t] ** 0.5
                b = (1 - alphas_cumprod[t]) ** 0.5
                real_sigma = b / a 
            t_list.append(int(t*control_factor))       
            self.scaling_factor_list.append(alphas_cumprod[int(t*control_factor)]**0.5)
        assert len(target_sigma_list)==len(t_list)==len(self.scaling_factor_list)

        return t_list # [t] 
    
    def forward(self, x, t_list, only_denoise=False, noise_add=True):
        if noise_add:
            noise = torch.randn_like(x) * self.sigma
            x_noised = x + noise # add noise [0~1] image space
        else:
            x_noised = x
        x_noised = x_noised * 2 -1 # normalize 
        x_noised = self.scaling_factor_list[0].item()*x_noised
        imgs = self.denoise(x_noised, t_list)
        imgs = (imgs / 2 + 0.5).clamp(0, 1) # convert [0~1]
        if only_denoise:
            return imgs
        else:
            with torch.set_grad_enabled(self.compute_attack):
                out = self.classifier(imgs)

        return out
    
    def denoise(self, x_t_start, timesteps):
        # scheduler setting
        self.scheduler.set_timesteps(timesteps=timesteps, device=x_t_start.device)
        timesteps = self.scheduler.timesteps
        
        B = x_t_start.shape[0] 
        prompt_embeds = self.prompt_embeds.repeat(B, 1, 1).to(x_t_start.device) # [B, 77, 4096]
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = self.negative_prompt_embeds.repeat(B, 1, 1).to(x_t_start.device)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        upscaled = x_t_start # set upscaled image to x_t_start 
        correct_factor = 1.8 # 1.8 is the best value for denoising. See ablation study in our paper.
        noise_level = torch.tensor([timesteps[0]*correct_factor] * upscaled.shape[0], device=upscaled.device) # noise_level
        condition = noise_level > self.scheduler.config.num_train_timesteps
        indices = condition.nonzero()
        noise_level[indices] = self.scheduler.config.num_train_timesteps        
        if self.do_classifier_free_guidance:
            noise_level = torch.cat([noise_level] * 2)

        # 1step denoising
        with torch.set_grad_enabled(self.compute_attack):
            for _, t in enumerate(timesteps):
                model_input = torch.cat([x_t_start, upscaled], dim=1) 
                model_input = torch.cat([model_input] * 2) if self.do_classifier_free_guidance else model_input
                model_input = self.scheduler.scale_model_input(model_input, t) # DDPM scheduler는 아무것도 안함.
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=noise_level,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]
                    
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                noise_pred, _ = noise_pred.split(x_t_start.shape[1], dim=1)
                x_t_start = self.scheduler.step(noise_pred, t, x_t_start, return_dict=True)['pred_original_sample']
        
        return x_t_start # torch.tensor & [-1~1]

    def save_image(self, x, t_list, savedir=None, savename=None):
        # save input
        x_pil = to_pil_image(x[0])
        x_pil.save(os.path.join(savedir,f'{savename}_input.png'))
        
        noise = torch.randn_like(x, device=f'cuda:{x.get_device()}') * self.sigma # 0.25 is noise
        x_noised = x + noise 
        
        # save noised input
        x_noised_pil = to_pil_image((x_noised[0]).clamp(0, 1))
        x_noised_pil.save(os.path.join(savedir, f'{savename}_noised.png'))
        
        x_noised = x_noised * 2 -1
        '''x = x * 2 - 1
        noise = torch.randn_like(x, device=f'cuda:{x.get_device()}') * self.sigma # 0.25 is noise
        x_noised = x + noise 
        # save noised input
        x_noised_pil = to_pil_image((x_noised[0]/ 2 + 0.5).clamp(0, 1))
        x_noised_pil.save(os.path.join(savedir, f'{savename}_noised.png'))'''

        x_noised = self.scaling_factor_list[0].item()*x_noised
        imgs = self.denoise(x_noised, t_list)
        # save denoised input x``
        imgs_pil = to_pil_image((imgs[0] / 2 + 0.5).clamp(0, 1))
        imgs_pil.save(os.path.join(savedir, f'{savename}_denoised.png'))                
