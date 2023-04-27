from transformers import DeiTModel, ViTModel, ResNetModel, AutoImageProcessor
from torchvision.models.resnet import resnet50


def get_backbone(backbone):
    """
    Args:
        backbone: name of the pretrained model that will be used as backbone 
    """
    if backbone == "deit":
        backbone = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
        image_processor = None

    elif backbone == "resnet50":
        # backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
        # image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50"
        backbone = resnet50(weights='ResNet50_Weights.DEFAULT')
        image_processor = None

    elif backbone == "dino":
        backbone = ViTModel.from_pretrained('facebook/dino-vits16')
        image_processor = None

    elif backbone == "resnet50_dino":
        backbone = ResNetModel.from_pretrained('Ramos-Ramos/dino-resnet-50')
        image_processor = None

    else:
        raise ValueError(f'{backbone} is not an backbone option.')
    
    return backbone, image_processor