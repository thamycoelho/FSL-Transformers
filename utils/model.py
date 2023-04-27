from transformers import DeiTModel, ViTModel, ResNetModel, AutoImageProcessor
from torchvision.models.resnet import resnet50
import torch


def get_backbone(backbone):
    """
    Args:
        backbone: name of the pretrained model that will be used as backbone 
    """
    if backbone == "deit":
        backbone = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

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