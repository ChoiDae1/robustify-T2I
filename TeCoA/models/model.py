import torch
import torch.nn as nn

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize(X):
    mu = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(X.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(X.device)
    return (X - mu) / std


def clip_img_preprocessing(X, img_size=224):
    X = torch.nn.functional.interpolate(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)
    return X


# for multiGPU clip -> 안씀
def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image, prompt_token=None):
        return self.model.encode_image(image, prompt_token)
     
     
def multiGPU_CLIP_image_logits(images, model, text_tokens, prompter=None, add_prompter=None):
    image_tokens = clip_img_preprocessing(images)
    prompt_token = None if add_prompter is None else add_prompter()
    if prompter is not None:
        image_tokens = prompter(image_tokens)
    return multiGPU_CLIP(None, None, model, image_tokens, text_tokens, prompt_token=prompt_token)[0]


def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)
    img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
    logits_per_image = img_embed @ scale_text_embed.t()
    #print(logits_per_image.shape)
    logits_per_text = scale_text_embed @ img_embed.t()
    #print(logits_per_text.shape)
    return logits_per_image, logits_per_text
