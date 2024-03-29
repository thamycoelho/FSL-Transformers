from transformers import DeiTModel, ViTModel, ResNetModel, AutoImageProcessor
from torchvision.models.resnet import resnet50, resnet18
import torch
import torch.nn as nn
from timm.scheduler import create_scheduler

from .vision_transformer_attn import vit_small

class SelfAttnPool(nn.Module):
    def __init__(self, feature_dim, units=None, do_the_sum=True, **kwargs):
        """
        Layer initialisation
        :param units: define the embedding dimension. If not specified (default),
                      it will be set to feat dimension.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.units = units if units else feature_dim
        self.do_the_sum = do_the_sum
        self.feature_dim = feature_dim

        weights_w = torch.zeros(self.feature_dim, self.units)
        nn.init.normal_(weights_w)
        self.W = nn.Parameter(weights_w)

        self.b = nn.Parameter(torch.zeros(self.units))

        weights_v =  torch.zeros(self.units, 1)
        nn.init.uniform_(weights_v)
        self.V = nn.Parameter(weights_v)
       
    
    def forward(self, x):
        """
        ui = tanh(xW+b)
        a = softmax(uV)
        o = sum(a*x)
        :param x: input tensor [batch_size, time_step, feat_len]
        :return: output tensor [batch_size, feat_len]
        """    
        units = self.units
        feature_dim = self.feature_dim
                
        # u = tanh(xW+b)
        u = torch.tanh(torch.matmul(x, self.W) + self.b)
        
        # a = softmax(uV)
        a = torch.softmax(torch.matmul(u, self.V), dim=1)
        
        o = x * a
        if self.do_the_sum:
            o = torch.sum(o, dim=1)
            
        return o
    
def get_output_dim(backbone, n_way, n_shot):
    if backbone in ['resnet50', 'resnet50_dino', 'resnet18']:
        feat_dim = 1000
    elif backbone in ['dino', 'deit_small', 'vit_mini']:
        feat_dim = 384
    elif backbone == 'deit':
        feat_dim = 768

    return (n_way, n_shot, feat_dim)

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
        
    elif backbone == "resnet18":
        backbone = resnet18(weights='ResNet18_Weights.DEFAULT')
    
    elif backbone == "vit_mini": # ViT pre-trained on miniImageNet
        backbone = vit_small(patch_size=16)
        checkpoint_file = 'utils/mini_imagenet/checkpoint1250.pth'

        state_dict = torch.load(checkpoint_file, map_location="cpu")['student']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError(f'{backbone} is not an backbone option.')
    
    return backbone


def get_aggregator(aggregator_name, support_features_shape):
    """
    Args:
        backbone: name of the aggregator to be used to generate prototypes
    """
    if aggregator_name == 'average':
        aggregator_func = torch.mean
    
    elif aggregator_name == 'max':
        aggregator_func = torch.max

    elif aggregator_name == 'log_sum_exp':
        aggregator_func = torch.logsumexp
    
    elif aggregator_name == 'lp_pool':
        kernel_size = (support_features_shape[-2], 1)
        aggregator_func = nn.LPPool2d(2, kernel_size)

    elif aggregator_name == 'self_attn':
        aggregator_func = SelfAttnPool(support_features_shape[-1])

    return aggregator_func
        
def apply_aggregator(aggregator, aggregator_name, support_features):
    if aggregator_name in ['average', 'log_sum_exp']:
        prototypes = aggregator(support_features, dim=1).unsqueeze(0)
    
    elif aggregator_name == 'max':
        prototypes, _ = aggregator(support_features, dim=1)
        prototypes = prototypes.unsqueeze(0)
    
    elif aggregator_name == 'lp_pool':
        prototypes = aggregator(support_features).view(1, support_features.shape[0], -1)

    elif aggregator_name == 'self_attn':
        prototypes = aggregator(support_features).unsqueeze(0)

    return prototypes
        
def get_optimizer(args, model):
    lr_scheduler = None

    # Define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Define scheduler
    if args.sched == 'cosine':
        lr_scheduler, _ = create_scheduler(args, optimizer)

    elif args.sched == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epochs, gamma=args.decay_rate)

    elif args.sched == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)

    return optimizer, lr_scheduler
