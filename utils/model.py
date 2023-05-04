from transformers import DeiTModel, ViTModel, ResNetModel, AutoImageProcessor
from torchvision.models.resnet import resnet50
import torch
import torch.nn as nn


def get_backbone(backbone):
    """
    Args:
        backbone: name of the pretrained model that will be used as backbone 
    """
    if backbone == "deit":
        backbone = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    
    elif backbone == "deit_small":
        backbone = DeiTModel.from_pretrained("facebook/deit-small-distilled-patch16-224")

    elif backbone == "resnet50":
        backbone = resnet50(weights='ResNet50_Weights.DEFAULT')

    elif backbone == "dino":
        backbone = ViTModel.from_pretrained('facebook/dino-vits16')

    elif backbone == "resnet50_dino":
        backbone = resnet50(weights='ResNet50_Weights.DEFAULT')

        state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
        backbone.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f'{backbone} is not an backbone option.')
    
    return backbone


def get_aggregator(aggregator, support_features):
    """
    Args:
        backbone: name of the aggregator to be used to generate prototypes
    """
    if aggregator == 'average':
        prototypes = torch.mean(support_features, dim=1).unsqueeze(0)
    
    elif aggregator == 'max':
        prototypes, _ = torch.max(support_features, dim=1)
        prototypes = prototypes.unsqueeze(0)

    elif aggregator == 'log_sum_exp':
        prototypes = torch.logsumexp(support_features, dim=1).unsqueeze(0)
    
    elif aggregator == 'lp_pool':
        kernel_size = (support_features.shape[-2], 1)
        pooling = nn.LPPool2d(2,kernel_size)
        prototypes = pooling(support_features).view(1, support_features.shape[0], -1)

    return prototypes
        