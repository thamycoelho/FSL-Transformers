from transformers import DeiTModel, ViTModel, ResNetModel


def get_backbone(backbone):
    """
    Args:
        backbone: name of the pretrained model that will be used as backbone 
    """
    if backbone == "deit":
        backbone = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

    elif backbone == "resnet50":
        backbone = ResNetModel.from_pretrained("microsoft/resnet-50")

    elif backbone == "dino":
        backbone = ViTModel.from_pretrained('facebook/dino-vits8')

    elif backbone == "resnet50_dino":
        backbone = ResNetModel.from_pretrained('Ramos-Ramos/dino-resnet-50')

    else:
        raise ValueError(f'{backbone} is not an backbone option.')