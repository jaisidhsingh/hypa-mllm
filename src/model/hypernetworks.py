import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import *

from src.model.model_utils import MLP
		

class HyperNetwork(nn.Module):
    def __init__(
        self, 
        param_shapes: List[List[int]], 
        cond_emb_dim: int, 
        num_cond_embs: int, 
        image_embed_dims: List[int], 
        hidden_layer_factors: List[int], 
        rescale_factor: Optional[float] = 10.0
    ) -> None:
        super().__init__()
        self.image_embed_dims = image_embed_dims
        self.param_shapes = param_shapes
        self.hidden_layer_factors = hidden_layer_factors

        self.cond_embs = nn.Embedding(num_cond_embs, cond_emb_dim)
        self.shape_embs = nn.Embedding(len(image_embed_dims), cond_emb_dim)

        self.to_weight = MLP(
            cond_emb_dim,
            [f * cond_emb_dim for f in self.hidden_layer_factors], 
            param_shapes[0][0] * param_shapes[0][1]
        )
        self.to_bias = MLP(
            cond_emb_dim, 
            [f * cond_emb_dim for f in self.hidden_layer_factors], 
            param_shapes[1][0]
        )
        if rescale_factor != 0.0:
            self.rescale_weight_prediction_params(rescale_factor)
        
        self.rescale_factor = rescale_factor     

    def rescale_weight_prediction_params(self, rescale_factor: float) -> None:
        # rescale the `weight` tensor data by `scale_factor` and set the bias to 0
        num_layers = self.to_weight.num_layers
        for i in [-1]:
            if hasattr(self.to_weight.layers[i], "weight") and hasattr(self.to_weight.layers[i], "bias"):
                self.to_weight.layers[i].bias.data.fill_(0.)
                self.to_weight.layers[i].weight.data /= rescale_factor

            if hasattr(self.to_bias.layers[i], "weight") and hasattr(self.to_bias.layers[i], "bias"):
                self.to_bias.layers[i].weight.data /= rescale_factor
                self.to_bias.layers[i].bias.data.fill_(0.)

        print("Rescaled parameters of `self.to_weight` and `self.to_bias`.")

    def compute_loss(self, logit_scale: Tensor, image_features: Tensor, text_features: Tensor) -> Tensor:
        logit_scale = logit_scale.exp().to(image_features.device)
        
        batch_size = image_features.shape[0]
        num_mappers = text_features.shape[0]

        labels = torch.arange(batch_size, dtype=torch.long).to(image_features.device).unsqueeze(0)
        labels = labels.repeat((num_mappers, 1))
        
        image_features = torch.permute(image_features, (1, 0, 2))
        logits1 = logit_scale * torch.einsum("nbd,ncd->nbc", image_features, text_features)

        preds = [logits1[i, :, :].argmax(dim=-1) for i in range(num_mappers)]
        corrects = [(preds[i] == labels[i, :]).sum().item() for i in range(num_mappers)]

        logits2 = logit_scale * torch.einsum("nbd,ncd->nbc", text_features, image_features)

        loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels))/2
        return loss.mean(), corrects

    def map_features(self, weights: Tensor, biases: Tensor, features: Tensor) -> Tensor:
        batch_size = features.shape[0]
        x = torch.einsum("nit,bt->nbi", weights, features)
        x = x + biases.unsqueeze(1).repeat((1, batch_size, 1))
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def forward(self, cond_id: Union[int, List[int]], image_embed_dim: int, normalize_output: bool = False) -> Tuple[Tensor, Tensor]:
        if type(cond_id) != list:
            cond_id = [cond_id]

        cond_id = torch.tensor(cond_id).long().to(self.cond_embs.weight.device) 
        num_conds = len(cond_id)
        cond_emb = self.cond_embs(cond_id) # shape: [num_conds, cond_emb_dim]
        if num_conds == 1:
            cond_emb = cond_emb.unsqueeze(0)

        shape_id = torch.tensor([self.image_embed_dims.index(image_embed_dim)]).long().to(self.cond_embs.weight.device)
        shape_emb = self.shape_embs(shape_id) # shape: [1, cond_emb_dim]
        shape_emb = shape_emb.repeat((num_conds, 1)) # shape: [num_conds, cond_emb_dim]

        final_cond_emb = cond_emb + shape_emb

        # predict mappers
        pred_weight = self.to_weight(final_cond_emb) # shape [num_conds, flattened_weight_dim]
        pred_weight = pred_weight.view((num_conds, self.param_shapes[0][0], self.param_shapes[0][1]))

        pred_bias = self.to_bias(final_cond_emb)
        pred_bias = pred_bias.view((num_conds, self.param_shapes[0][0]))

        pred_weight = pred_weight[:, :image_embed_dim, :]
        pred_bias = pred_bias[:, :image_embed_dim]

        if normalize_output:
            pred_weight = pred_weight * (1 / pred_weight[0].numel()) ** 0.5
            pred_bias = pred_bias * (1 / pred_bias[0].numel()) ** 0.5
        
        return pred_weight, pred_bias
