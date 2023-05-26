import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from utils import get_backbone, get_aggregator, apply_aggregator, get_output_dim

class ProtoNet(nn.Module):
    def __init__(self, scale=False):
        super().__init__()

        self.scale = scale

        if scale:
            self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
            self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
        
    def forward(self, support, query, mode="cos_sim"):
        support = F.normalize(support, p=2, dim=support.dim()-1, eps=1e-12)
        query = F.normalize(query, p=2, dim=query.dim()-1, eps=1e-12)
        
        if mode == "cos_sim":
            score = query @ support.transpose(1, 2)

        if self.scale:
            score = self.scale_cls * (score + self.bias)
            
        return score
        
        
class DeiTForFewShot(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        # self.num_labels = config.num_labels
        # self.deit = DeiTModel(config, add_pooling_layer=False)
        self.backbone_name = args.backbone
        self.aggregator_name = args.aggregator
        self.backbone = get_backbone(self.backbone_name)
        
        self.aggregator = get_aggregator(self.aggregator_name, get_output_dim(self.backbone_name, args.nClsEpisode, args.nSupport))
        # Classifier 
        self.classifier = ProtoNet(scale=args.scale_score)
        
    def forward(
        self,
        query: Optional[torch.Tensor] = None,
        support: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None,
    ):
        
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        """        
        n_way = support_labels.max() + 1
        
        B, nSupp, C, H, W = support.shape

        support = support.view(-1, C, H, W)
        query =  query.view(-1, C, H, W)

        # Get support features
        support_features = self.backbone(
            support,
        )

        if self.backbone_name in ['deit', 'dino', 'deit_small']:
            support_features = support_features[0][:,0,:]
        support_features = support_features.view(B, nSupp, -1)
        
        # Get prototypes (avg pooling)
        sorted_support_labels = torch.sort(support_labels)
        support_features = F.embedding(sorted_support_labels.indices.view(n_way, torch.div(support.shape[0], n_way, rounding_mode='trunc')), support_features.squeeze())
        prototypes = apply_aggregator(self.aggregator, self.aggregator_name, support_features)

        # Get query features
        query_features = self.backbone(
            query,
        )
        
        if self.backbone_name in ['deit', 'dino', 'deit_small']:
            query_features = query_features[0][:,0,:]
        query_features = query_features.view(B, query.shape[0], -1)

        logits = self.classifier(support=prototypes, query=query_features)
        
        return logits