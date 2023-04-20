from transformers import DeiTPreTrainedModel, DeiTConfig, DeiTFeatureExtractor, DeiTModel, AutoModel, AutoConfig
from transformers.modeling_outputs import ImageClassifierOutput
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import requests
import matplotlib.pyplot as plt
from typing import Optional, Set, Tuple, Union

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
    def __init__(self) -> None:
        super().__init__()
        
        # self.num_labels = config.num_labels
        # self.deit = DeiTModel(config, add_pooling_layer=False)
        self.deit = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", add_pooling_layer=False)
        
        # Classifier 
        self.classifier = ProtoNet()
        
    def forward(
        self,
        query: Optional[torch.Tensor] = None,
        support: Optional[torch.Tensor] = None,
        support_labels: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ):
        
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        n_way = support_labels.max() + 1
        
        B, nSupp, C, H, W = support.shape

        # Get support features
        support_features = self.deit(
            support.view(-1, C, H, W),
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        support_features = support_features[0][:,0,:]
        support_features = support_features.view(B, nSupp, -1)
        
        # Get prototypes (avg pooling)
        sorted_support_labels = torch.sort(support_labels)
        support_features = F.embedding(sorted_support_labels.indices.view(n_way, torch.div(support.shape[1], n_way, rounding_mode='trunc')), support_features.squeeze())
        prototypes = torch.mean(support_features, dim=1).unsqueeze(0)

        # Get query featurhttps://www.linkedin.com/in/camilaherculano/es 
        query_features = self.deit(
            query.view(-1, C, H, W),
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        query_features = query_features[0][:,0,:]
        query_features = query_features.view(B, query.shape[1], -1)

        logits = self.classifier(support=prototypes, query=query_features)
        
        return logits