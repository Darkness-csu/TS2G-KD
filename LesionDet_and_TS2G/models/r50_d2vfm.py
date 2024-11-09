# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone



class R50_D2VFM(nn.Module):

    def __init__(self, backbone, num_feature_levels, needle=1, out_feat_dim=768, adapter_type='mpa'):
        #needle表示从R50第几层的输出取结果，这决定了后续特征的dim
        super().__init__()
        assert num_feature_levels in [1,2,3,4], "Invalid value for num_feature_levels" 
        assert needle in [0,1,2] and needle < num_feature_levels, 'Invalid value for needle'
        assert adapter_type in ['p', 'm', 'ma', 'mp', 'mpa'], 'Invalid value for adapter_type'
        
        self.dim_choices = {0:512, 1:1024, 2:2048}
        self.needle = needle

        self.backbone = backbone
        self.input_dim = self.dim_choices[self.needle]
        self.out_dim = out_feat_dim
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.adapter_type = adapter_type

        if self.adapter_type == 'p':
            self.adapter = Adapter_P(input_dim=self.input_dim, output_dim=self.out_dim)
        elif self.adapter_type == 'm':
            self.adapter = Adapter_M(input_dim=self.input_dim, output_dim=self.out_dim)
        elif self.adapter_type == 'ma':
            self.adapter = Adapter_M(input_dim=self.input_dim, output_dim=self.out_dim, skip_connect=True) 
        elif self.adapter_type == 'mp':
            self.adapter = Adapter_MP(input_dim=self.input_dim, output_dim=self.out_dim)
        elif self.adapter_type == 'mpa':
            self.adapter = Adapter_MP(input_dim=self.input_dim, output_dim=self.out_dim, skip_connect=True)
        else:
            self.adapter = nn.Identity()
        
        #init adapter
        for n, m in self.adapter.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, samples: NestedTensor):
        #new
        # print(samples.tensors.shape)
        # exit()
        # samples, feats = samples[0], torch.stack(samples[1])
        # print(type(samples.tensors),samples.tensors.shape) #(B,3,H,W)
        # print(type(feats), feats.shape) #(B,H,W)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, _ = self.backbone(samples)
        
        #new
        assert self.needle < len(features)
        src_t, _ = features[self.needle].decompose()
        inner_feat = self.avgpool(src_t).view(src_t.shape[0], -1)
        out_feat = self.adapter(inner_feat)
        #new

        out = { 'adapted_feat':out_feat}
        return out
    
    def extract_adapted_feat(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, _ = self.backbone(samples)
        assert self.needle < len(features)
        src_t, _ = features[self.needle].decompose()
        inner_feat = self.avgpool(src_t).view(src_t.shape[0], -1)
        out_feat = self.adapter(inner_feat)
        return out_feat
    
   
class Adapter_MP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, mlp_ratio=0.25, act_layer=nn.ReLU, skip_connect=False):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(input_dim * mlp_ratio)
        self.act = act_layer()
        self.mlp_fc1 = nn.Linear(input_dim, hidden_dim)
        self.mlp_fc2 = nn.Linear(hidden_dim, input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.mlp_fc1(self.act(x))
        xs = self.mlp_fc2(self.act(xs))
        if self.skip_connect:
            x = self.act(x + xs)
        else:
            x = self.act(xs)
        out = self.proj(x)
        return out
    
class Adapter_M(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, mlp_ratio=0.25, act_layer=nn.ReLU, skip_connect=False):
        super().__init__()
        self.skip_connect = skip_connect
        hidden_dim = int(input_dim * mlp_ratio)
        self.act = act_layer()
        self.mlp_fc1 = nn.Linear(input_dim, hidden_dim)
        self.mlp_fc2 = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.mlp_fc1(self.act(x))
        
        if self.skip_connect:
            xs = self.act(x + xs)
        else:
            xs = self.act(xs)

        out = self.mlp_fc2(xs)
        
        return out
    
class Adapter_P(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, act_layer=nn.ReLU):
        super().__init__()
        
        self.act = act_layer()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        
        out = self.proj(self.act(x))

        return out
    
class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, construct_func):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.construct_func = construct_func

    def loss_construct(self, outputs, feats, loss_type):
        adapted_feat = outputs['adapted_feat']
        bs,_ = adapted_feat.shape
        if loss_type == 'MSE':
            # loss_construct = F.mse_loss(adapted_feat, torch.stack(feats), reduction='sum') #直接全部求和应该有问题，需要取均值
            # loss_construct = F.mse_loss(adapted_feat, torch.stack(feats), reduction='mean') #直接mean相当于是取每个emb的单个维度的平均mse损失
            loss_construct = torch.sum(F.mse_loss(adapted_feat, torch.stack(feats), reduction='none'), dim=-1)
            loss_construct = torch.mean(loss_construct)
        elif loss_type == 'MAE':
            # loss_construct = F.l1_loss(adapted_feat, torch.stack(feats), reduction='sum')
            loss_construct = torch.sum(F.l1_loss(adapted_feat, torch.stack(feats), reduction='none'), dim=-1)
            loss_construct = torch.mean(loss_construct)
        elif loss_type == 'Cosine':
            adapted_feat_normalized = F.normalize(adapted_feat, p=2, dim=-1)
            feats_normalized = F.normalize(torch.stack(feats), p=2, dim=-1)
            cosine_similarity = torch.sum(adapted_feat_normalized * feats_normalized, dim=-1)
            loss_construct = torch.mean(1 - cosine_similarity)
        else:
            raise ValueError(f'loss function {loss_type} not supported')
        
        losses = {'loss_construct':loss_construct}
        return losses
    
    
    def forward(self, outputs, feats):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        losses.update(self.loss_construct(outputs, feats, self.construct_func))
        losses['class_error']=torch.tensor(-1).to(feats[0].device)
        return losses



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_r50_d2vfm(args):
    #needle=1, out_feat_dim=768, adapter_type='mpa'
    num_classes = 10 + 1
    device = torch.device(args.device)
    backbone = build_backbone(args)

    distill_dataset = args.distill_dataset
    if distill_dataset in ['biomedclip', 'dinov2', 'mae']:
        out_feat_dim = 768
    elif distill_dataset in ['clip', 'plip']:
        out_feat_dim = 512
    elif distill_dataset == 'gigapath':
        out_feat_dim = 1536
    else:
        raise ValueError(f'Invalid value for arg distill_dataset:{distill_dataset}')
    
    model = R50_D2VFM(
        backbone,
        num_feature_levels=args.num_feature_levels,
        needle=args.needle,
        out_feat_dim = out_feat_dim,
        adapter_type = args.adapter_type
    )
    
    weight_dict = {'loss_construct': args.construct_loss_coef}
    
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, weight_dict, args.construct_func)
    criterion.to(device)

    return model, criterion
