from typing import List, Tuple

import torch
import open_clip


def build_openclip_model(model_name: str,
                         pretrained: str,
                         device: str
                         ) -> Tuple[torch.nn.Module, object, object]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.eval().to(device)
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_prompts(model: torch.nn.Module,
                   tokenizer,
                   prompts: List[str],
                   device: str
                   ) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = _l2_normalize(text_features)
    return text_features


@torch.no_grad()
def encode_images(model: torch.nn.Module,
                  images: torch.Tensor
                  ) -> torch.Tensor:
    image_features = model.encode_image(images)
    image_features = _l2_normalize(image_features)
    return image_features


def compute_similarity(image_features: torch.Tensor,
                       text_features: torch.Tensor,
                       logit_scale: torch.Tensor = None
                       ) -> torch.Tensor:
    scores = image_features @ text_features.transpose(0, 1)
    if logit_scale is not None:
        scores = scores * logit_scale.exp()
    return scores


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)