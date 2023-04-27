import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from utils import get_backbone

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, support, query, mode="cos_sim"):
        support = F.normalize(support, p=2, dim=support.dim()-1, eps=1e-12)
        query = F.normalize(query, p=2, dim=query.dim()-1, eps=1e-12)
        
        if mode == "cos_sim":
            score = query @ support.transpose(1, 2)

        return score
        
        
class DeiTForFewShot(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()
        
        # self.num_labels = config.num_labels
        # self.deit = DeiTModel(config, add_pooling_layer=False)
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone)

        # Classifier 
        self.classifier = ProtoNet()
        
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
        # Prepare data
        if self.image_processor:
            support = self.image_processor(support)
            query = self.image_processor(query)

        # Get support features
        support_features = self.backbone(
            support,
        )

        if self.backbone_name in ['deit', 'dino']:
            support_features = support_features[0][:,0,:]
        support_features = support_features.view(B, nSupp, -1)
        
        # Get prototypes (avg pooling)
        sorted_support_labels = torch.sort(support_labels)
        support_features = F.embedding(sorted_support_labels.indices.view(n_way, torch.div(support.shape[0], n_way, rounding_mode='trunc')), support_features.squeeze())
        prototypes = torch.mean(support_features, dim=1).unsqueeze(0)

        # Get query featurhttps://www.linkedin.com/in/camilaherculano/es 
        query_features = self.backbone(
            query,
        )
        
        if self.backbone_name in ['deit', 'dino']:
            query_features = query_features[0][:,0,:]
        query_features = query_features.view(B, query.shape[0], -1)

        logits = self.classifier(support=prototypes, query=query_features)
        
        return logits